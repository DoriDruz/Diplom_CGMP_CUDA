// Итерац. МСГ с предобуславливателем на процессоре[, многоядерном процессоре и видеокарте]
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>

#include <windows.h>

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif
using namespace std;

//global
const int S = 134862;

//global GPU
//меняемые значения
const int block_size = 1;
const int grid_size = 1; 

const int per_thread = S / (grid_size * block_size);

void add_nevyazka(double nev) {
	ofstream nevyazka;
	nevyazka.open("nevyazka.dat", ios_base::app);
	nevyazka << nev << endl;
	nevyazka.close();
}

void clear_nevyazka() {
	//clear file with nevyazka
	ofstream nevyazka_file;
	nevyazka_file.open("nevyazka.dat", ofstream::trunc);
	nevyazka_file.close();
}

void write_in_file(double *vec, double size, string name) {
	cout << "Begin write in file" << endl;
	ofstream result_file;
	result_file.open(name, ofstream::trunc); //X.dat

	for (int i = 0; i < size; ++i) {
		result_file << setprecision(16) << vec[i] << endl;
	}

	result_file.close();

	cout << "End write in file: " << endl;
}

void write_rA_in_file(double *vec) {
	cout << "Begin write in file" << endl;
	ofstream result_file;
	result_file.open("rA.dat", ofstream::trunc);

	for (int i = 1; i < S * 7 + 1; ++i) {
		result_file << setprecision(16) << vec[i - 1] << " ";
		if (i % 7 == 0) {
			result_file << endl;
		}
	}

	result_file.close();

	cout << "End write in file: " << endl;
}

void create_matr(ifstream& file, double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		file >> setprecision(16) >> matr[i];
	}
}

void copy_vec(double *matr_one, double *matr_two) {
	for (int i = 0; i < S; ++i) {
		matr_one[i] = matr_two[i];
	}
}

//-------------------------------------------------------------------------------------------

void matr_on_vec(double *matr, double *vec, double *res_vec) {
	int second_count = 0;
	int third_count = 0;
	int fourth_count = 0;
	int fifth_count = 0;
	int sixth_count = 0;

	for (int i = 0; i < S; ++i) {
		if (i == 0) {
			res_vec[i] = \
				matr[0] * vec[0] \
				+ matr[1] * vec[1] \
				+ matr[2] * vec[247] \
				+ matr[3] * vec[494];
		}
		else if (i > 0 && i < 247) {
			res_vec[i] = \
				matr[7 * i] * vec[second_count] \
				+ matr[7 * i + 1] * vec[second_count + 1] \
				+ matr[7 * i + 2] * vec[second_count + 2] \
				+ matr[7 * i + 3] * vec[second_count + 248] \
				+ matr[7 * i + 4] * vec[second_count + 495];
			second_count++;
		}
		else if (i > 246 && i < 494) {
			res_vec[i] = \
				matr[7 * i] * vec[third_count] \
				+ matr[7 * i + 1] * vec[third_count + 246] \
				+ matr[7 * i + 2] * vec[third_count + 247] \
				+ matr[7 * i + 3] * vec[third_count + 248] \
				+ matr[7 * i + 4] * vec[third_count + 494] \
				+ matr[7 * i + 5] * vec[third_count + 741];
			third_count++;
		}
		else if (i > 493 && i < 134368) {
			res_vec[i] = \
				matr[7 * i] * vec[fourth_count] \
				+ matr[7 * i + 1] * vec[fourth_count + 247] \
				+ matr[7 * i + 2] * vec[fourth_count + 493] \
				+ matr[7 * i + 3] * vec[fourth_count + 494] \
				+ matr[7 * i + 4] * vec[fourth_count + 495] \
				+ matr[7 * i + 5] * vec[fourth_count + 741] \
				+ matr[7 * i + 6] * vec[fourth_count + 988];
			fourth_count++;
		}
		else if (i > 134367 && i < 134615) {
			res_vec[i] = \
				matr[7 * i + 1] * vec[fifth_count + 133874] \
				+ matr[7 * i + 2] * vec[fifth_count + 134121] \
				+ matr[7 * i + 3] * vec[fifth_count + 134367] \
				+ matr[7 * i + 4] * vec[fifth_count + 134368] \
				+ matr[7 * i + 5] * vec[fifth_count + 134369] \
				+ matr[7 * i + 6] * vec[fifth_count + 134615];
			fifth_count++;
		}
		else if (i > 134614 && i < 134861) {
			res_vec[i] = \
				matr[7 * i + 2] * vec[sixth_count + 134121] \
				+ matr[7 * i + 3] * vec[sixth_count + 134368] \
				+ matr[7 * i + 4] * vec[sixth_count + 134614] \
				+ matr[7 * i + 5] * vec[sixth_count + 134615] \
				+ matr[7 * i + 6] * vec[sixth_count + 134616];
			sixth_count++;
		}
		else if (i == 134861) {
			res_vec[i] = \
				matr[7 * i + 3] * vec[134367] \
				+ matr[7 * i + 4] * vec[134614] \
				+ matr[7 * i + 5] * vec[134860] \
				+ matr[7 * i + 6] * vec[134861];
		}
	}
}

__global__ void GPU_matr_on_vec(double *matr, double *vec, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	int second_count = 0;
	int third_count = 0;
	int fourth_count = 0;
	int fifth_count = 0;
	int sixth_count = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		if (i == 0) {
			res_vec[i] = \
				__dmul_rn(matr[0], vec[0]) \
				+ __dmul_rn(matr[1], vec[1]) \
				+ __dmul_rn(matr[2], vec[247]) \
				+ __dmul_rn(matr[3], vec[494]);
		}
		else if (i > 0 && i < 247) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i], vec[second_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[second_count + 1]) \
				+ __dmul_rn(matr[7 * i + 2], vec[second_count + 2]) \
				+ __dmul_rn(matr[7 * i + 3], vec[second_count + 248]) \
				+ __dmul_rn(matr[7 * i + 4], vec[second_count + 495]);

			second_count++;
		}
		else if (i > 246 && i < 494) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i], vec[third_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[third_count + 246]) \
				+ __dmul_rn(matr[7 * i + 2], vec[third_count + 247]) \
				+ __dmul_rn(matr[7 * i + 3], vec[third_count + 248]) \
				+ __dmul_rn(matr[7 * i + 4], vec[third_count + 494]) \
				+ __dmul_rn(matr[7 * i + 5], vec[third_count + 741]);

			third_count++;
		}
		else if (i > 493 && i < 134368) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i], vec[fourth_count]) \
				+ __dmul_rn(matr[7 * i + 1], vec[fourth_count + 247]) \
				+ __dmul_rn(matr[7 * i + 2], vec[fourth_count + 493]) \
				+ __dmul_rn(matr[7 * i + 3], vec[fourth_count + 494]) \
				+ __dmul_rn(matr[7 * i + 4], vec[fourth_count + 495]) \
				+ __dmul_rn(matr[7 * i + 5], vec[fourth_count + 741]) \
				+ __dmul_rn(matr[7 * i + 6], vec[fourth_count + 988]);

			fourth_count++;
		}
		else if (i > 134367 && i < 134615) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 1], vec[fifth_count + 133874]) \
				+ __dmul_rn(matr[7 * i + 2], vec[fifth_count + 134121]) \
				+ __dmul_rn(matr[7 * i + 3], vec[fifth_count + 134367]) \
				+ __dmul_rn(matr[7 * i + 4], vec[fifth_count + 134368]) \
				+ __dmul_rn(matr[7 * i + 5], vec[fifth_count + 134369]) \
				+ __dmul_rn(matr[7 * i + 6], vec[fifth_count + 134615]);

			fifth_count++;
		}
		else if (i > 134614 && i < 134861) {
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 2], vec[sixth_count + 134121]) \
				+ __dmul_rn(matr[7 * i + 3], vec[sixth_count + 134368]) \
				+ __dmul_rn(matr[7 * i + 4], vec[sixth_count + 134614]) \
				+ __dmul_rn(matr[7 * i + 5], vec[sixth_count + 134615]) \
				+ __dmul_rn(matr[7 * i + 6], vec[sixth_count + 134616]);

			sixth_count++;
		}
		else if (i == 134861) { //last_element_position = 944034
			res_vec[i] = \
				__dmul_rn(matr[7 * i + 3], vec[134367]) \
				+ __dmul_rn(matr[7 * i + 4], vec[134614]) \
				+ __dmul_rn(matr[7 * i + 5], vec[134860]) \
				+ __dmul_rn(matr[7 * i + 6], vec[134861]);
		}
	}
	/*for (int i = 0; i < 1; ++i) {
	printf("res_vec[i]: %.14f =\n matr[0] %.14f * vec[0] %.14f \n + matr[1] %.14f * vec[1] %.14f \n + matr[2] %.14f * vec[247] %.14f \n + matr[3] %.14f * vec[494] %.14f \n",
	res_vec[i], matr[0], vec[0], matr[1], vec[1], matr[2], vec[247], matr[3], vec[494]);
	}*/
	//printf("res_vec[i]: %.14f =\n matr[0] %.14f * vec[0] %.14f \n + matr[1] %.14f * vec[1] %.14f \n + matr[2] %.14f * vec[247] %.14f \n + matr[3] %.14f * vec[494] %.14f \n",
	//	res_vec[0], matr[0], vec[0], matr[1], vec[1], matr[2], vec[247], matr[3], vec[494]);
}

//-------------------------------------------------------------------------------------------


double vec_on_vec(double *vec_one, double *vec_two) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += vec_one[i] * vec_two[i];
	}

	return res;
}

__global__ void GPU_vec_on_vec(double *vec_one, double *vec_two, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;
	*res = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		tmp_res_per_thread += __dmul_rn(vec_one[i], vec_two[i]);
	}
	atomicAdd(res, tmp_res_per_thread);
	//printf("%.14f = %.14f = %.14f + %.14f\n", *res, tmp_res_per_thread, vec_one[0], vec_two[0]);
}

//-------------------------------------------------------------------------------------------


void vec_on_num(double *vec, double num, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec[i] * num;
	}
}

//-------------------------------------------------------------------------------------------

void sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] + vec_two[i];
	}
}

__global__ void GPU_sum_vec(double *vec_one, double *vec_two, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = __dadd_ru(vec_one[i], vec_two[i]);
	}
}

//-------------------------------------------------------------------------------------------

void dif_vec(double *vec_one, double *vec_two, double *res_vec) {
	for (int i = 0; i < S; ++i) {
		res_vec[i] = vec_one[i] - vec_two[i];
	}
}

__global__ void GPU_dif_vec(double *vec_one, double *vec_two, double *res_vec) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		res_vec[i] = __dsub_ru(vec_one[i], vec_two[i]);
		//debug
		/*if (i > S / 2 && i < S / 2 + 10) {
		printf("%.14f = %.14f - %.14f\n", res_vec[i], vec_one[i], vec_two[i]);
		}*/
	}
}

//-------------------------------------------------------------------------------------------

void show_vec(double *matr, double size) {
	for (int i = 0; i < size; ++i) {
		cout << matr[i] << "; ";
	}
	cout << endl;
}

__global__ void GPU_show_vec(double *matr, double size) {
	printf("X: %.16f\n", matr[0]);
}

//-------------------------------------------------------------------------------------------

double norm_vec(double *vec) {
	double res = 0;
	for (int i = 0; i < S; ++i) {
		res += vec[i] * vec[i];
	}
	return sqrt(res);
}

__global__ void GPU_norm_vec(double *vec, double *res) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	double tmp_res_per_thread = 0;
	*res = 0;

	for (int i = thread * per_thread; i < (thread + 1) * per_thread; ++i) {
		*res += __dmul_rn(vec[i], vec[i]); 
	}
	*res = __dsqrt_ru(*res);
}

//-------------------------------------------------------------------------------------------

void nullify(double *vec) {
	for (int i = 0; i < S; ++i) {
		vec[i] = 0;
	}
}

void make_rA(double *A, double *C) {
	for (int i = 0; i < S; ++i) {
		for (int j = 0; j < 7; ++j) {
			A[7 * i + j] *= (1 / C[i]);
		}
	}
	//write_rA_in_file(A);
}

//-------------------------------------------------------------------------------------------

void copy_matr(double *m_one, double *m_two) {
	for (int i = 0; i < S * 7; ++i) {
		m_one[i] = m_two[i];
	}
}

__global__ void GPU_copy_matr(double *m_one, double *m_two) {
	int thread = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = (thread * per_thread) * 7; i < ((thread + 1) * per_thread) * 7; ++i) {
		m_one[i] = m_two[i];
	}
}

//-------------------------------------------------------------------------------------------

__global__ void GPU_ak_bk(double *up, double *down, double *ak_bk) {
	//printf("up and down: %f, %f\n", *up, *down);
	//*ak_bk = __ddiv_rn(*up, *down);
	*ak_bk = __ddiv_ru(*up, *down);
	//printf("ak_bk: %.16f\n", *ak_bk);
}

__global__ void GPU_check_nev(double *up, double *down, double *eps, double Eps, double *x_res) {
	*eps = __ddiv_ru(*up, *down);
	printf("Nev: %.16f %s %.2f\n\n", *eps, ((*eps < Eps) ? "<" : ">"), Eps);
	/*if (*eps < Eps) {
	return;
	}*/
}

//-------------------------------------------------------------------------------------------


void CGMP(double *A, double *F, double *C, double *vec_C, clock_t begin_algo) {
	//Условие останова. Эпсилон
	const double Eps = 0.1;

	//Оригинальная матрица A
	double *or_A = new double[S * 7];

	//Вектора
	double *x = new double[S];
	double *r = new double[S];
	double *p = new double[S];
	double *z = new double[S];

	double *r_k1 = new double[S];
	double *x_k1 = new double[S];
	double *p_k1 = new double[S];
	double *z_k1 = new double[S];

	//Коэффициенты
	double ak = 0;
	double bk = 0;

	//tmp
	double *tmp = new double[S];
	double *Apk = new double[S];
	double *stop_tmp = new double[S];
	double *stop_tmp_b = new double[S];
	double ak_up = 0;
	double ak_down = 0;
	double bk_up = 0;
	double bk_down = 0;
	double stop_num = 1;
	double stop_up = 0;
	double stop_down = 0;
	int count_tmp = 1;
	double stop_less = INFINITY;
	double stop_max = NULL;

	//to device from arguments
	double *dev_A = NULL;
	double *dev_F = NULL;

	double *dev_x = NULL;
	double *dev_r = NULL;
	double *dev_p = NULL;
	double *dev_z = NULL;

	double *dev_ak = NULL;
	double *dev_bk;

	double *dev_tmp = NULL;
	double *dev_Apk = NULL;
	double *dev_r_k1 = NULL;
	double *dev_x_k1 = NULL;
	double *dev_p_k1 = NULL;
	double *dev_z_k1 = NULL;
	double *dev_up = 0;
	double *dev_down = 0;

	//for stop_deal
	double stop_up = 0;
	double stop_down = 0;
	double stop_eps = 1;
	double *stop_vec = new double[S];
	double *stop_tmp = new double[S];

	double *dev_stop_up;
	double *dev_stop_down;
	double *dev_stop_eps;
	double *dev_stop_tmp = NULL;
	double *dev_stop_vec = NULL;

	CUDA_CALL(cudaMalloc(&dev_A, S * 7 * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_F, S * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_x, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_z, S * sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_ak, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_bk, sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_Apk, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_r_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_x_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_p_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_z_k1, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_up, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_down, sizeof(double)));

	CUDA_CALL(cudaMalloc(&dev_stop_up, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_down, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_eps, sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_tmp, S * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_stop_vec, S * sizeof(double)));

	//Заполняем все вектора нулями
	for (int i = 0; i < S; ++i) {
		x[i] = r[i] = p[i] = z[i] = 0;
		stop_tmp_b[i] = stop_tmp[i] = Apk[i] = tmp[i] = r_k1[i] = x_k1[i] = p_k1[i] = z_k1[i] = 0;
	}

	CUDA_CALL(cudaMemcpy(dev_A, A, S * 7 * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_F, F, S * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_x, x, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r, r, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p, p, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_z, z, S * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_ak, &ak, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_bk, &bk, sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_Apk, Apk, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r_k1, r_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_x_k1, x_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p_k1, p_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_p_k1, z_k1, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_tmp, tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_up, &ak_up, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_down, &ak_down, sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMemcpy(dev_stop_up, &stop_up, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_down, &stop_down, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_eps, &stop_eps, sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_tmp, stop_tmp, S * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_stop_vec, stop_vec, S * sizeof(double), cudaMemcpyHostToDevice));

	//Подготовка перед циклом
	//or_A = A;
	copy_matr(or_A, A);
	//rA = C^(-1) * A
	make_rA(A, vec_C);
	//r0 = b - A*x0		|x0 = 0 -> A*x0 = 0 -> r0 = b|  |r0 = C*b - A*x0?|
	//copy_vec(r, F);
	//matr_on_vec(C, F, r);
	GPU_matr_on_vec << <block_size, grid_size >> >(C, dev_F, dev_r);
	//p0 = C*r0
	//matr_on_vec(C, r, p);
	GPU_matr_on_vec << <block_size, grid_size >> >(C, dev_r, dev_p);
	//z0 = p0
	copy_vec(z, p);

	while (!(stop_num < Eps)) {
		cout << count_tmp << " / " << S << endl;
		clock_t begin_CGMR = clock();

		//ak = (r_k, z_k) / (A*p_k, p_k)
		//(r_k, z_k)
		//ak_up = vec_on_vec(r, z);
		GPU_vec_on_vec << <block_size, grid_size >> >(dev_r, dev_z, dev_up);
		//A*p_k
		//matr_on_vec(A, p, Apk);
		GPU_matr_on_vec << <block_size, grid_size >> >(dev_A, dev_p, dev_Apk);
		//(A*p_k, p_k)
		//ak_down = vec_on_vec(Apk, p);
		GPU_vec_on_vec << <block_size, grid_size >> >(dev_Apk, dev_p, dev_down);
		//ak = ...
		//ak = ak_up / ak_down;
		GPU_ak_bk << <block_size, grid_size >> >(dev_up, dev_down, dev_up);

		//x_k1 = x_k + ak*p_k
		//ak*p_k
		vec_on_num(p, ak, tmp);
		//x_k1 = x_k + ...
		//sum_vec(x, tmp, x_k1);
		GPU_sum_vec << <block_size, grid_size >> >(dev_x, dev_tmp, dev_x_k1);

		nullify(tmp);
		//r_k1 = r_k - ak*A*p_k
		//ak*A*p_k
		vec_on_num(Apk, ak, tmp);
		//r_k1 = r_k - ...
		//dif_vec(r, tmp, r_k1);
		GPU_dif_vec << <block_size, grid_size >> >(dev_r, dev_tmp, dev_r_k1);

		//z_k1 = C*r_k1
		//matr_on_vec(C, r_k1, z_k1);
		GPU_matr_on_vec << <block_size, grid_size >> >(C, dev_r_k1, dev_z_k1);

		//bk = (r_k1, z_k1) / (r_k, z_k)
		//(r_k1, z_k1)
		//bk_up = vec_on_vec(r_k1, z_k1);
		GPU_vec_on_vec << <block_size, grid_size >> >(dev_r_k1, dev_z_k1, dev_up);

		//(r_k, z_k)
		//bk_down = vec_on_vec(r, z);
		GPU_vec_on_vec << <block_size, grid_size >> >(dev_r, dev_z, dev_down);
		//bk = ...
		//bk = bk_up / bk_down;
		GPU_ak_bk << <block_size, grid_size >> >(dev_up, dev_down, dev_up);

		//p_k1 = z_k1 + bk*p_k
		//bk*p_k
		nullify(tmp);
		vec_on_num(p, bk, tmp);
		//p_k1 = z_k1 + ...
		//sum_vec(z_k1, tmp, p_k1);
		GPU_sum_vec << <block_size, grid_size >> >(dev_z_k1, dev_tmp, dev_p_k1);

		//Показываем первые 10 решений на каждой итерации
		cout << "X[0]: ";
		show_vec(x_k1, 1);

		//Условие останова - ДОБАВИТЬ ВМЕСТО b | C * b?
		// ||A*z_k - b|| / ||b|| < Eps
		//A*z_k
		nullify(tmp);
		//matr_on_vec(A, x_k1, tmp);
		GPU_matr_on_vec << <block_size, grid_size >> >(dev_A, dev_x_k1, dev_stop_tmp);
		//C*b
		//matr_on_vec(C, F, stop_tmp_b);
		GPU_matr_on_vec << <block_size, grid_size >> >(C, dev_F, dev_stop_tmp);
		//A*z_k - b
		//dif_vec(tmp, stop_tmp_b, stop_tmp);
		GPU_dif_vec << <block_size, grid_size >> >(dev_stop_tmp, dev_F, dev_stop_vec);
		//||A*z_k - b||
		//stop_up = norm_vec(stop_tmp);
		GPU_norm_vec << <block_size, grid_size >> >(dev_stop_vec, dev_stop_up
		//||b||
		//stop_down = norm_vec(stop_tmp_b);
		GPU_norm_vec << <block_size, grid_size >> >(dev_F, dev_stop_down);
		//||..|| / ||..||
		//stop_num = stop_up / stop_down;
		GPU_check_nev << <1, 1 >> >(dev_stop_up, dev_stop_down, dev_stop_eps, Eps, dev_x_k1);

		if (stop_num < stop_less) {
			stop_less = stop_num;
		}
		else if (stop_num > stop_max) {
			stop_max = stop_num;
		}

		//Показываем невязку
		CUDA_CALL(cudaMemcpy(&stop_eps, dev_stop_eps, sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaDeviceSynchronize());
		cout << "Diff: " << stop_num << " " << ((stop_num < Eps) ? "<" : ">") << " " << Eps << " | L: " << stop_less << " M: " << stop_max << endl << endl;
		add_nevyazka(stop_num);

		if (stop_num < Eps) {
			CUDA_CALL(cudaMemcpy(x_k1, dev_x_k1, S * sizeof(double), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaDeviceSynchronize());
			write_in_file(x_k1, S, "X_2.dat");
		}

		//Копируем вектора
		/*copy_vec(x, x_k1);
		copy_vec(r, r_k1);
		copy_vec(p, p_k1);
		copy_vec(z, z_k1);*/

		CUDA_CALL(cudaMemcpy(dev_p, dev_p_k1, S * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(dev_x, dev_x_k1, S * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(dev_r, dev_r_k1, S * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(dev_z, dev_z_k1, S * sizeof(double), cudaMemcpyDeviceToDevice));

		//Доп условие для остановки
		//if (count_tmp == S / 16) {
		//	write_in_file(x_k1, S, "X.dat");
		//	break;
		//}
		++count_tmp;

		//Очищаем вектора
		nullify(x_k1);
		nullify(r_k1);
		nullify(p_k1);
		nullify(z_k1);
		nullify(stop_tmp);
		nullify(stop_tmp_b);
		nullify(tmp);
		nullify(Apk);

		clock_t end_CGMR = clock();

		double CGMR_time = double(end_CGMR - begin_CGMR) / CLOCKS_PER_SEC;
		//cout << "Iteration runtime: " << CGMR_time << endl << endl;

		clock_t end_algo = clock();
		double algo_time = double(end_algo - begin_algo) / CLOCKS_PER_SEC;
		//cout << "Algoritm runtime: " << algo_time << endl;

	}

	//Очищаем память
	delete(x);
	delete(r);
	delete(p);
	delete(z);
	delete(x_k1);
	delete(r_k1);
	delete(p_k1);
	delete(z_k1);
	delete(tmp);
	delete(Apk);
	delete(stop_tmp);
	delete(stop_tmp_b);

	cudaFree(dev_x);
	cudaFree(dev_r);
	cudaFree(dev_p);
	cudaFree(dev_z);
	cudaFree(dev_x_k1);
	cudaFree(dev_r_k1);
	cudaFree(dev_p_k1);
	cudaFree(dev_z_k1);
	cudaFree(dev_tmp);
	cudaFree(dev_Apk);
	cudaFree(dev_A);
	cudaFree(dev_F);
	cudaFree(dev_ak);
	cudaFree(dev_bk);
	cudaFree(dev_up);
	cudaFree(dev_down);
	cudaFree(dev_stop_up);
	cudaFree(dev_stop_down);
	cudaFree(dev_stop_eps);
	cudaFree(dev_stop_tmp);
	cudaFree(dev_stop_vec);
}

void main() {
	//A priori C^-1 = C

	clock_t begin_algo = clock();

	double *matr_A = new double[S * 7];
	double *matr_F = new double[S];
	double *matr_C = new double[S * 7];
	double *vec_C = new double[S];

	ifstream A;
	ifstream F;
	ifstream C;
	ifstream A3;

	clock_t begin = clock();

	A.open("A_with_01.dat");		// 134862 * 7	| A.dat
	F.open("F.dat");				// 134862		| F.dat
	C.open("C.dat");				// 134862 * 7	| C.dat
	A3.open("A3.dat");				// 134862		| F.dat

	create_matr(A, matr_A, S * 7);
	create_matr(F, matr_F, S);
	create_matr(C, matr_C, S * 7);
	create_matr(A3, vec_C, S);

	A.close();
	F.close();
	C.close();
	A3.close();

	clear_nevyazka();
	CGMP(matr_A, matr_F, matr_C, vec_C, begin_algo);

	clock_t end = clock();
	double time = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Runtime: " << time << endl;

	delete(matr_A);
	delete(matr_F);
	delete(matr_C);
	delete(vec_C);

	system("pause");
}