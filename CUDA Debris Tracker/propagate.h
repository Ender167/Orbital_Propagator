#pragma once
#include <math.h>

constexpr float Re = 6378.137f;
constexpr float J2 = 1.08262668e-3f;
constexpr float J2_Re2 = J2 * Re * Re;

float specific_energy_j2_cpu(
    float x, float y, float z,
    float vx, float vy, float vz,
    float mu)
{
    float r2 = x * x + y * y + z * z;
    r2 = fmaxf(r2, 1e-6f);
    float inv_r = 1.0f / sqrtf(r2);

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

void propagate_omp(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* out_x, float* out_y, float* out_z,
    float* out_en, float dt, float mu, int N, int steps, int output_interval)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {

        float x = pos_x[i];
        float y = pos_y[i];
        float z = pos_z[i];

        float vx = vel_x[i];
        float vy = vel_y[i];
        float vz = vel_z[i];

        int out_idx = 0;

        float E0 = specific_energy_j2_cpu(x, y, z, vx, vy, vz, mu);

        for (int t = 0; t < steps; t++) {

            float r2 = x * x + y * y + z * z;

            r2 = fmaxf(r2, 1e-6f);
            float inv_r = 1.0f / sqrtf(r2);

            inv_r = inv_r * (1.5f - 0.5f * r2 * inv_r * inv_r);

            float inv_r2 = inv_r * inv_r;
            float inv_r3 = inv_r2 * inv_r;

            float factor = -mu * inv_r3;

            float z2 = z * z;
            float r2_inv = inv_r2;

            float k = 1.5f * J2 * mu * (Re * Re) * inv_r3 * r2_inv;

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
                float E = specific_energy_j2_cpu(
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
}

void propagate_cpu(
    float* pos_x, float* pos_y, float* pos_z,
    float* vel_x, float* vel_y, float* vel_z,
    float* out_x, float* out_y, float* out_z,
    float* out_en, float dt, float mu, int N, int steps, int output_interval)
{
    for (int i = 0; i < N; i++) {

        float x = pos_x[i];
        float y = pos_y[i];
        float z = pos_z[i];

        float vx = vel_x[i];
        float vy = vel_y[i];
        float vz = vel_z[i];

        int out_idx = 0;

        float E0 = specific_energy_j2_cpu(x, y, z, vx, vy, vz, mu);

        for (int t = 0; t < steps; t++) {

            float r2 = x * x + y * y + z * z;

            r2 = fmaxf(r2, 1e-6f);
            float inv_r = 1.0f / sqrtf(r2);

            inv_r = inv_r * (1.5f - 0.5f * r2 * inv_r * inv_r);

            float inv_r2 = inv_r * inv_r;
            float inv_r3 = inv_r2 * inv_r;

            float factor = -mu * inv_r3;

            float z2 = z * z;
            float r2_inv = inv_r2;

            float k = 1.5f * J2 * mu * (Re * Re) * inv_r3 * r2_inv;

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

                float E = specific_energy_j2_cpu(
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
}