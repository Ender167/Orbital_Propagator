#include <fstream>
#include <vector>
#include <string>

struct Vec3 {
    float x, y, z;
};
struct Vec3D {
    double x, y, z;
};
struct Header {
    int num_frames;
    int num_particles;
};

class BinaryWriter {
    std::ofstream file;
    Header header;
    int frames_written = 0;

public:
    BinaryWriter(const std::string& filename, int total_frames, int N) {
        header.num_frames = total_frames;
        header.num_particles = N;

        file.open(filename, std::ios::binary);
        file.write(reinterpret_cast<char*>(&header), sizeof(Header));
    }

    void write_frame(int t, int N, float* pos_x, float* pos_y, float* pos_z) {

        file.write(reinterpret_cast<char*>(&t), sizeof(int));

        std::vector<Vec3> buffer(N);
        for (int i = 0; i < N; i++) {
            buffer[i] = { pos_x[i], pos_y[i], pos_z[i] };
        }

        file.write(reinterpret_cast<char*>(buffer.data()), N * sizeof(Vec3));

        frames_written++;
    }
    void write_frame(int t, int N, double* pos_x, double* pos_y, double* pos_z) {

        file.write(reinterpret_cast<char*>(&t), sizeof(int));

        std::vector<Vec3D> buffer(N);
        for (int i = 0; i < N; i++) {
            buffer[i] = { pos_x[i], pos_y[i], pos_z[i] };
        }

        file.write(reinterpret_cast<char*>(buffer.data()), N * sizeof(Vec3D));

        frames_written++;
    }

    void close() {
        file.close();
    }
};
