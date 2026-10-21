
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>
#include "BinaryWriter.h"
#include "propagate.h"
#include "SimulationData.h"

using namespace std;
using namespace chrono;



void write_positions(int t, int N, float* pos_x, float* pos_y, float* pos_z) {
    string fileName = "E:\\Projects\\Disertation\\Python_Visualization\\frames\\frame_" + to_string(t) + ".csv";
    ofstream file(fileName);
    for (int i = 0; i < N; i++) {
        file << pos_x[i] << "," << pos_y[i] << "," << pos_z[i] << "\n";
    }
    file.close();
}

void write_frames(int N, int num_outputs, int output_interval, BinaryWriter& writer, float* h_out_x, float* h_out_y, float* h_out_z) {

    auto start_writing = high_resolution_clock::now();

    int num_threads = omp_get_max_threads();
    if (num_outputs < num_threads) {
        omp_set_num_threads(num_outputs);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_outputs; i++) {
        writer.write_frame(i * output_interval, N, &h_out_x[i * N], &h_out_y[i * N], &h_out_z[i * N]);
    }

    auto end_writing = high_resolution_clock::now();
    milliseconds t_writing = duration_cast<milliseconds>(end_writing - start_writing);

    cout << "Writing (ms): " << t_writing.count() << " ms" << endl;
}
void write_frames(int N, int num_outputs, int output_interval, BinaryWriter& writer, double* h_out_x, double* h_out_y, double* h_out_z) {

    auto start_writing = high_resolution_clock::now();
    int num_threads = omp_get_max_threads();
    if (num_outputs < num_threads * 2) {
        omp_set_num_threads(num_outputs);
    }

    #pragma omp parallel for
    for (int i = 0; i < num_outputs; i++) {
        writer.write_frame(i * output_interval, N, &h_out_x[i * N], &h_out_y[i * N], &h_out_z[i * N]);
    }
    auto end_writing = high_resolution_clock::now();
    milliseconds t_writing = duration_cast<milliseconds>(end_writing - start_writing);

    cout << "Writing (ms): " << t_writing.count() << " ms" << endl;
}

void compute_metrics(float* energy_errors, int N, int num_outputs)
{
    float max_err = 0.0f;
    double rms_accum = 0.0;
    long long count = 0;

    for (int s = 0; s < num_outputs; s++) {
        int base = s * N;

        for (int p = 0; p < N; p++) {
            float e = energy_errors[base + p];

            max_err = std::max(max_err, e);

            rms_accum += (double)e * (double)e;
            count++;
        }
    }

    double rms_err = std::sqrt(rms_accum / count);

    int last_base = (num_outputs - 1) * N;

    float max_final_err = 0.0f;
    double mean_final_err = 0.0;

    for (int p = 0; p < N; p++) {
        float e = energy_errors[last_base + p];

        max_final_err = std::max(max_final_err, e);
        mean_final_err += e;
    }

    mean_final_err /= N;

    printf("\n=== Energy Error Metrics ===\n");
    printf("Max error (all):      %.9e\n", max_err);
    printf("RMS error (all):      %.9e\n", rms_err);
    printf("Max final error:      %.9e\n", max_final_err);
    printf("Mean final error:     %.9e\n", mean_final_err);
    printf("============================\n");
}
void compute_metrics(double* energy_errors, int N, int num_outputs)
{
    double max_err = 0.0f;
    double rms_accum = 0.0;
    long long count = 0;

    for (int s = 0; s < num_outputs; s++) {
        int base = s * N;

        for (int p = 0; p < N; p++) {
            double e = energy_errors[base + p];

            max_err = std::max(max_err, e);

            rms_accum += e * e;
            count++;
        }
    }

    double rms_err = std::sqrt(rms_accum / count);

    int last_base = (num_outputs - 1) * N;

    double max_final_err = 0.0f;
    double mean_final_err = 0.0;

    for (int p = 0; p < N; p++) {
        double e = energy_errors[last_base + p];

        max_final_err = std::max(max_final_err, e);
        mean_final_err += e;
    }

    mean_final_err /= N;

    printf("\n=== Energy Error Metrics ===\n");
    printf("Max error (all):      %.9e\n", max_err);
    printf("RMS error (all):      %.9e\n", rms_err);
    printf("Max final error:      %.9e\n", max_final_err);
    printf("Mean final error:     %.9e\n", mean_final_err);
    printf("============================\n");
}
void read_debris(std::string filename, double* h_pos_x, double* h_pos_y, double* h_pos_z, double* h_vel_x, double* h_vel_y, double* h_vel_z) {

    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "file not found" << endl;
        return;
    }

    int n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    double* buffer = new double[6];
    for (int i = 0; i < n; i++) {
        file.read(reinterpret_cast<char*>(buffer), 6 * sizeof(double));

        h_pos_x[i] = buffer[0];
        h_pos_y[i] = buffer[1];
        h_pos_z[i] = buffer[2];

        h_vel_x[i] = buffer[3];
        h_vel_y[i] = buffer[4];
        h_vel_z[i] = buffer[5];

    }
    file.close();
}
void read_debris(std::string filename, float* h_pos_x, float* h_pos_y, float* h_pos_z, float* h_vel_x, float* h_vel_y, float* h_vel_z) {

    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cout << "file not found" << endl;
        return;
    }

    int n;
    file.read(reinterpret_cast<char*>(&n), sizeof(n));
    float* buffer = new float[6];
    for (int i = 0; i < n; i++) {
        file.read(reinterpret_cast<char*>(buffer), 6 * sizeof(float));

        h_pos_x[i] = buffer[0];
        h_pos_y[i] = buffer[1];
        h_pos_z[i] = buffer[2];

        h_vel_x[i] = buffer[3];
        h_vel_y[i] = buffer[4];
        h_vel_z[i] = buffer[5];

    }
    file.close();
}
void generate_debris(std::string filename, int N, float mu) {

    std::ofstream file;

    float pi = 3.14159265f;
    float R_earth = 6371.0f;  // km
    float altitude = 400.0f;  // km
    float r_mag = R_earth + altitude;

    float v_circ = sqrt(mu / r_mag);

    file.open(filename, std::ios::binary);
    file.write(reinterpret_cast<char*>(&N), sizeof(int));

    for (int i = 0; i < N; i++) {
        float theta = 2.0f * pi * i / N;

        float u = (float)rand() / RAND_MAX;
        float v = (float)rand() / RAND_MAX;

        float inclination = acosf(2.0f * u - 1.0f);
        float raan = 2.0f * pi * v;

        float cos_i = cosf(inclination);
        float sin_i = sinf(inclination);
        float cos_O = cosf(raan);
        float sin_O = sinf(raan);

        float x_orb = r_mag * cosf(theta);
        float y_orb = r_mag * sinf(theta);

        float vx_orb = -v_circ * sinf(theta);
        float vy_orb = v_circ * cosf(theta);

        float* buffer = new float[6];

        buffer[0] = cos_O * x_orb - sin_O * cos_i * y_orb;
        buffer[1] = sin_O * x_orb + cos_O * cos_i * y_orb;
        buffer[2] = sin_i * y_orb;

        buffer[3] = cos_O * vx_orb - sin_O * cos_i * vy_orb;
        buffer[4] = sin_O * vx_orb + cos_O * cos_i * vy_orb;
        buffer[5] = sin_i * vy_orb;

        file.write(reinterpret_cast<char*>(buffer), 6 * sizeof(float));

        delete[] buffer;
    }
    file.close();
}
void generate_debris_d(std::string filename, int N, float mu) {

    std::ofstream file;

    double pi = 3.14159265f;
    double R_earth = 6371.0f;  // km
    double altitude = 400.0f;  // km
    double r_mag = R_earth + altitude;

    double v_circ = sqrt(mu / r_mag);

    file.open(filename, std::ios::binary);
    file.write(reinterpret_cast<char*>(&N), sizeof(int));

    for (int i = 0; i < N; i++) {
        double theta = 2.0f * pi * i / N;

        double u = (double)rand() / RAND_MAX;
        double v = (double)rand() / RAND_MAX;

        double inclination = acosf(2.0f * u - 1.0f);
        double raan = 2.0f * pi * v;

        double cos_i = cosf(inclination);
        double sin_i = sinf(inclination);
        double cos_O = cosf(raan);
        double sin_O = sinf(raan);

        double x_orb = r_mag * cosf(theta);
        double y_orb = r_mag * sinf(theta);

        double vx_orb = -v_circ * sinf(theta);
        double vy_orb = v_circ * cosf(theta);

        double* buffer = new double[6];

        buffer[0] = cos_O * x_orb - sin_O * cos_i * y_orb;
        buffer[1] = sin_O * x_orb + cos_O * cos_i * y_orb;
        buffer[2] = sin_i * y_orb;

        buffer[3] = cos_O * vx_orb - sin_O * cos_i * vy_orb;
        buffer[4] = sin_O * vx_orb + cos_O * cos_i * vy_orb;
        buffer[5] = sin_i * vy_orb;

        file.write(reinterpret_cast<char*>(buffer), 6 * sizeof(double));
    }
    file.close();
}

__device__ __inline__ double specific_energy_j2(
    double x, double y, double z,
    double vx, double vy, double vz,
    double mu)
{
    double r2 = x * x + y * y + z * z;
    r2 = fmax(r2, 1e-12);

    double inv_r = rsqrt(r2);

    inv_r = inv_r * (1.5 - 0.5 * r2 * inv_r * inv_r);

    double inv_r2 = inv_r * inv_r;
    double inv_r3 = inv_r2 * inv_r;

    double z2 = z * z;

    double v2 = vx * vx + vy * vy + vz * vz;
    double kinetic = 0.5 * v2;

    double U0 = -mu * inv_r;

    double Uj2 = 0.5 * mu * J2_Re2 * inv_r3 * (3.0 * z2 * inv_r2 - 1.0);

    return kinetic + U0 + Uj2;
}

__device__ __inline__ float specific_energy_j2(
    float x, float y, float z,
    float vx, float vy, float vz,
    float mu)
{
    float r2 = x * x + y * y + z * z;
    r2 = fmaxf(r2, 1e-6f);

    float inv_r = rsqrtf(r2);

    inv_r = inv_r * (1.5f - 0.5f * r2 * inv_r * inv_r);

    float inv_r2 = inv_r * inv_r;
    float inv_r3 = inv_r2 * inv_r;

    float z2 = z * z;

    float v2 = vx * vx + vy * vy + vz * vz;
    float kinetic = 0.5f * v2;

    float U0 = -mu * inv_r;

    float Uj2 = 0.5f * mu * J2_Re2 * inv_r3 * (3.0f * z2 * inv_r2 - 1.0f);

    return kinetic + U0 + Uj2;
}

__global__ void propagate(
    float* __restrict__ pos_x, float* __restrict__ pos_y, float* __restrict__ pos_z,
    float* __restrict__ vel_x, float* __restrict__ vel_y, float* __restrict__ vel_z,
    float* __restrict__ out_x, float* __restrict__ out_y, float* __restrict__ out_z,
    float* __restrict__ out_en, float dt, float mu, int N, int steps, int output_interval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = pos_x[i];
    float y = pos_y[i];
    float z = pos_z[i];

    float vx = vel_x[i];
    float vy = vel_y[i];
    float vz = vel_z[i];

    int out_idx = 0;

    float E0 = specific_energy_j2(x, y, z, vx, vy, vz, mu);
    
    for (int t = 0; t < steps; t++) {

        float r2 = x * x + y * y + z * z;

        r2 = fmaxf(r2, 1e-6f);
        float inv_r = rsqrtf(r2);

        inv_r = inv_r * (1.5f - 0.5f * r2 * inv_r * inv_r);

        float inv_r2 = inv_r * inv_r;
        float inv_r3 = inv_r2 * inv_r;

        float factor = -mu * inv_r3;

        float z2 = z * z;
        float r2_inv = inv_r2;


        float k = 1.5f * J2_Re2 * mu * inv_r3 * r2_inv;

        float f = 5.0f * z2 * r2_inv;

        float ax = factor * x + k * x * (f - 1.0f);
        float ay = factor * y + k * y * (f - 1.0f);
        float az = factor * z + k * z * (f - 3.0f);

        vx = fmaf(ax, dt, vx);
        vy = fmaf(ay, dt, vy);
        vz = fmaf(az, dt, vz);

        x = fmaf(vx, dt, x);
        y = fmaf(vy, dt, y);
        z = fmaf(vz, dt, z);

        if (t % output_interval == 0) {
            int idx = out_idx * N + i;

            float E = specific_energy_j2(
                x, y, z,
                vx, vy, vz,
                mu);

            float rel_energy_error = fabsf((E - E0) / E0);

            out_x[idx] = x;
            out_y[idx] = y;
            out_z[idx] = z;

            out_en[idx] = rel_energy_error;
            out_idx++;
        }
    }

    pos_x[i] = x;
    pos_y[i] = y;
    pos_z[i] = z;

    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}

__global__ void propagate(
    double* __restrict__ pos_x, double* __restrict__ pos_y, double* __restrict__ pos_z,
    double* __restrict__ vel_x, double* __restrict__ vel_y, double* __restrict__ vel_z,
    double* __restrict__ out_x, double* __restrict__ out_y, double* __restrict__ out_z,
    double* __restrict__ out_en,
    double dt, double mu, int N, int steps, int output_interval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double x = pos_x[i];
    double y = pos_y[i];
    double z = pos_z[i];

    double vx = vel_x[i];
    double vy = vel_y[i];
    double vz = vel_z[i];

    int out_idx = 0;

    double E0 = specific_energy_j2 (x, y, z, vx, vy, vz, mu);

    for (int t = 0; t < steps; t++) {

        double r2 = x * x + y * y + z * z;

        r2 = fmax(r2, 1e-6);

        double inv_r = rsqrt(r2);

        inv_r = inv_r * (1.5 - 0.5 * r2 * inv_r * inv_r);

        double inv_r2 = inv_r * inv_r;
        double inv_r3 = inv_r2 * inv_r;

        double factor = -mu * inv_r3;

        double z2 = z * z;
        double r2_inv = inv_r2;

        double k = 1.5 * J2_Re2 * mu * inv_r3 * r2_inv;

        double f = 5.0 * z2 * r2_inv;

        double ax = factor * x + k * x * (f - 1.0);
        double ay = factor * y + k * y * (f - 1.0);
        double az = factor * z + k * z * (f - 3.0);

        vx = fma(ax, dt, vx);
        vy = fma(ay, dt, vy);
        vz = fma(az, dt, vz);

        x = fma(vx, dt, x);
        y = fma(vy, dt, y);
        z = fma(vz, dt, z);

        if (t % output_interval == 0) {
            int idx = out_idx * N + i;

            double E = specific_energy_j2(x, y, z, vx, vy, vz, mu);

            double rel_energy_error = fabs((E - E0) / E0);

            out_x[idx] = x;
            out_y[idx] = y;
            out_z[idx] = z;

            out_en[idx] = rel_energy_error;

            out_idx++;
        }
    }

    pos_x[i] = x;
    pos_y[i] = y;
    pos_z[i] = z;

    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}


int main(int argc, char *argv[])
{
    if (argc != 3) {
        return 1;
    }
    //0 - seq, 1 - openmp, 2 - gpu
    int version_to_run = static_cast<int>(strtoul(argv[1], nullptr, 10));
    if (version_to_run < 0 || version_to_run > 2)
        version_to_run = 2;

    int precision = static_cast<int>(strtoul(argv[2], nullptr, 10));

    if (precision < 0 || precision > 1)
        precision = 0;

    int N = 100000;
    float dt = 0.1f; // seconds
    float mu = 3.986e5f; // km^3/s^2
    
    int steps = 20000;
    int output_interval = 10;

    int num_outputs = (steps - 1) / output_interval + 1;

    SimulationData<float> sim_f(N, num_outputs);
    SimulationData<double> sim_d(N, num_outputs);

    //generate_debris_d("E:\\Projects\\Disertation\\Python_Visualization\\datasetBinary\\debris" + to_string(N) + "d.bin", N, mu);

    auto start_read = high_resolution_clock::now();
    if(precision == 0)
        read_debris("E:\\Projects\\Disertation\\Python_Visualization\\datasetBinary\\debris" + to_string(N) + ".bin", sim_f.h_pos_x, sim_f.h_pos_y, sim_f.h_pos_z, sim_f.h_vel_x, sim_f.h_vel_y, sim_f.h_vel_z);
    if(precision == 1)
        read_debris("E:\\Projects\\Disertation\\Python_Visualization\\datasetBinary\\debris" + to_string(N) + "d.bin", sim_d.h_pos_x, sim_d.h_pos_y, sim_d.h_pos_z, sim_d.h_vel_x, sim_d.h_vel_y, sim_d.h_vel_z);
    auto end_read = high_resolution_clock::now();
    milliseconds t_read = duration_cast<milliseconds>(end_read - start_read);

    cout << "Reading (ms): " << t_read.count() << " ms" << endl;


    if (version_to_run == 0) {
        //seq
        BinaryWriter writer("E:\\Projects\\Disertation\\Python_Visualization\\framesBinary\\simulation_seq.bin", num_outputs, N);

        cout << "---SEQUENTIAL---" << endl;
        auto start_exec = high_resolution_clock::now();
        propagate_cpu(sim_f.h_pos_x, sim_f.h_pos_y, sim_f.h_pos_z, sim_f.h_vel_x, sim_f.h_vel_y, sim_f.h_vel_z, sim_f.h_out_x, sim_f.h_out_y, sim_f.h_out_z, sim_f.h_out_en, dt, mu, N, steps, output_interval);
        auto end_exec = high_resolution_clock::now();
        milliseconds t_exec = duration_cast<milliseconds>(end_exec - start_exec);

        cout << "Execution (ms): " << t_exec.count() << " ms" << endl;

        write_frames(N, num_outputs, output_interval, writer, sim_f.h_out_x, sim_f.h_out_y, sim_f.h_out_z);
        cout << "---DONE SEQ---" << endl;
        compute_metrics(sim_f.h_out_en, N, num_outputs);

    }
    else if (version_to_run == 1) {
        // openmp
        BinaryWriter writer("E:\\Projects\\Disertation\\Python_Visualization\\framesBinary\\simulation_omp.bin", num_outputs, N);

        cout << "---OPENMP---" << endl;
        auto start_exec = high_resolution_clock::now();
        propagate_omp(sim_f.h_pos_x, sim_f.h_pos_y, sim_f.h_pos_z, sim_f.h_vel_x, sim_f.h_vel_y, sim_f.h_vel_z, sim_f.h_out_x, sim_f.h_out_y, sim_f.h_out_z, sim_f.h_out_en, dt, mu, N, steps, output_interval);
        auto end_exec = high_resolution_clock::now();
        milliseconds t_exec = duration_cast<milliseconds>(end_exec - start_exec);

        cout << "Execution (ms): " << t_exec.count() << " ms" << endl;

        write_frames(N, num_outputs, output_interval, writer, sim_f.h_out_x, sim_f.h_out_y, sim_f.h_out_z);
        cout << "---DONE OPENMP---" << endl;
        compute_metrics(sim_f.h_out_en, N, num_outputs);
    }
    else {
        // CUDA
        BinaryWriter writer("E:\\Projects\\Disertation\\Python_Visualization\\framesBinary\\simulation.bin", num_outputs, N);
        
        cout << "---CUDA---" << endl;

        if (precision == 0) {
            /*
            float* d_pos_x, * d_pos_y, * d_pos_z;
            float* d_vel_x, * d_vel_y, * d_vel_z;
            float* d_out_x, * d_out_y, * d_out_z, * d_out_en;

            cudaMalloc(&d_pos_x, N * sizeof(float));
            cudaMalloc(&d_pos_y, N * sizeof(float));
            cudaMalloc(&d_pos_z, N * sizeof(float));

            cudaMalloc(&d_vel_x, N * sizeof(float));
            cudaMalloc(&d_vel_y, N * sizeof(float));
            cudaMalloc(&d_vel_z, N * sizeof(float));

            cudaMalloc(&d_out_x, num_outputs * N * sizeof(float));
            cudaMalloc(&d_out_y, num_outputs * N * sizeof(float));
            cudaMalloc(&d_out_z, num_outputs * N * sizeof(float));
            cudaMalloc(&d_out_en, num_outputs * N * sizeof(float));
            */

            cudaMemcpy(sim_f.d_pos_x, sim_f.h_pos_x, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_f.d_pos_y, sim_f.h_pos_y, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_f.d_pos_z, sim_f.h_pos_z, N * sizeof(float), cudaMemcpyHostToDevice);

            cudaMemcpy(sim_f.d_vel_x, sim_f.h_vel_x, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_f.d_vel_y, sim_f.h_vel_y, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_f.d_vel_z, sim_f.h_vel_z, N * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            propagate << <blocks, threads >> > (
                sim_f.d_pos_x, sim_f.d_pos_y, sim_f.d_pos_z,
                sim_f.d_vel_x, sim_f.d_vel_y, sim_f.d_vel_z,
                sim_f.d_out_x, sim_f.d_out_y, sim_f.d_out_z, sim_f.d_out_en,
                dt, mu, N, steps, output_interval);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);


            cout << "Execution (ms): " << ms << " ms" << endl;

            cudaMemcpy(sim_f.h_out_x, sim_f.d_out_x, num_outputs * N * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(sim_f.h_out_y, sim_f.d_out_y, num_outputs * N * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(sim_f.h_out_z, sim_f.d_out_z, num_outputs * N * sizeof(float), cudaMemcpyDeviceToHost);

            cudaMemcpy(sim_f.h_out_en, sim_f.d_out_en, num_outputs * N * sizeof(float), cudaMemcpyDeviceToHost);

            write_frames(N, num_outputs, output_interval, writer, sim_f.h_out_x, sim_f.h_out_y, sim_f.h_out_z);

            cout << "---DONE CUDA---" << endl;

            compute_metrics(sim_f.h_out_en, N, num_outputs);

            /*
            cudaFree(d_pos_x);
            cudaFree(d_pos_y);
            cudaFree(d_pos_z);
            cudaFree(d_vel_x);
            cudaFree(d_vel_y);
            cudaFree(d_vel_z);

            cudaFree(d_out_x);
            cudaFree(d_out_y);
            cudaFree(d_out_z);
            */
        }
        else {

            cudaMemcpy(sim_d.d_pos_x, sim_d.h_pos_x, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_d.d_pos_y, sim_d.h_pos_y, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_d.d_pos_z, sim_d.h_pos_z, N * sizeof(double), cudaMemcpyHostToDevice);

            cudaMemcpy(sim_d.d_vel_x, sim_d.h_vel_x, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_d.d_vel_y, sim_d.h_vel_y, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(sim_d.d_vel_z, sim_d.h_vel_z, N * sizeof(double), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            propagate << <blocks, threads >> > (
                sim_d.d_pos_x, sim_d.d_pos_y, sim_d.d_pos_z,
                sim_d.d_vel_x, sim_d.d_vel_y, sim_d.d_vel_z,
                sim_d.d_out_x, sim_d.d_out_y, sim_d.d_out_z, sim_d.d_out_en,
                dt, mu, N, steps, output_interval);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);


            cout << "Execution (ms): " << ms << " ms" << endl;

            cudaMemcpy(sim_d.h_out_x, sim_d.d_out_x, num_outputs* N * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sim_d.h_out_y, sim_d.d_out_y, num_outputs* N * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sim_d.h_out_z, sim_d.d_out_z, num_outputs* N * sizeof(double), cudaMemcpyDeviceToHost);

            cudaMemcpy(sim_d.h_out_en, sim_d.d_out_en, num_outputs* N * sizeof(double), cudaMemcpyDeviceToHost);

            write_frames(N, num_outputs, output_interval, writer, sim_d.h_out_x, sim_d.h_out_y, sim_d.h_out_z);

            cout << "---DONE CUDA---" << endl;

            compute_metrics(sim_d.h_out_en, N, num_outputs);

        }
         

    }

    return 0;
}
