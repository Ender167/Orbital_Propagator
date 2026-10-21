template <typename T>
struct SimulationData {
    size_t N, num_outputs;

    T* h_pos_x, * h_pos_y, * h_pos_z;
    T* h_vel_x, * h_vel_y, * h_vel_z;
    T* h_out_x, * h_out_y, * h_out_z, * h_out_en;

    T* d_pos_x, * d_pos_y, * d_pos_z;
    T* d_vel_x, * d_vel_y, * d_vel_z;
    T* d_out_x, * d_out_y, * d_out_z, * d_out_en;

    SimulationData(size_t n, size_t nout)
        : N(n), num_outputs(nout)
    {
        h_pos_x = new T[N];
        h_pos_y = new T[N];
        h_pos_z = new T[N];

        h_vel_x = new T[N];
        h_vel_y = new T[N];
        h_vel_z = new T[N];

        h_out_x = new T[N * num_outputs];
        h_out_y = new T[N * num_outputs];
        h_out_z = new T[N * num_outputs];
        h_out_en = new T[N * num_outputs];


        cudaMalloc(&d_pos_x, N * sizeof(T));
        cudaMalloc(&d_pos_y, N * sizeof(T));
        cudaMalloc(&d_pos_z, N * sizeof(T));

        cudaMalloc(&d_vel_x, N * sizeof(T));
        cudaMalloc(&d_vel_y, N * sizeof(T));
        cudaMalloc(&d_vel_z, N * sizeof(T));

        cudaMalloc(&d_out_x, N * num_outputs * sizeof(T));
        cudaMalloc(&d_out_y, N * num_outputs * sizeof(T));
        cudaMalloc(&d_out_z, N * num_outputs * sizeof(T));
        cudaMalloc(&d_out_en, N * num_outputs * sizeof(T));
    }

    ~SimulationData() {
        delete[] h_pos_x;
        delete[] h_pos_y;
        delete[] h_pos_z;
        delete[] h_vel_x;
        delete[] h_vel_y;
        delete[] h_vel_z;
        delete[] h_out_x;
        delete[] h_out_y;
        delete[] h_out_z;
        delete[] h_out_en;

        cudaFree(d_pos_x);
        cudaFree(d_pos_y);
        cudaFree(d_pos_z);
        cudaFree(d_vel_x);
        cudaFree(d_vel_y);
        cudaFree(d_vel_z);
        cudaFree(d_out_x);
        cudaFree(d_out_y);
        cudaFree(d_out_z);
        cudaFree(d_out_en);
    }
};