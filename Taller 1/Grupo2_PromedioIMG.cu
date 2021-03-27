#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void readFile(int *L, int *M, int *N, float **R, float **G, float **B){
    FILE *fp = fopen("images6.txt", "r");
    
    fscanf(fp, "%d %d %d", L, M, N);
    int size = (*M)*(*N);
    int i, j;
    float *R1 = (float*)malloc(*L*size*sizeof(float));
    float *G1 = (float*)malloc(*L*size*sizeof(float));
    float *B1 = (float*)malloc(*L*size*sizeof(float));
    
    for(i = 0; i < *L; i++){
        for(j = 0; j < size; j++){
            fscanf(fp, "%f", &R1[j+i*size]); // OJO AHÍ
        }
        for(j = 0; j < size; j++){
            fscanf(fp, "%f", &G1[j+i*size]); // OJO AHÍ
        }
        for(j = 0; j < size; j++){
            fscanf(fp, "%f", &B1[j+i*size]); // OJO AHÍ
        }
    }

    fclose(fp);
    *R = R1; *G = G1; *B = B1;
}

void writeFile(int M, int N, float *R, float *G, float *B, char* name){
    FILE *fp = fopen(name, "w");
    fprintf(fp, "%d %d\n", M, N);
    int size = M*N;
    for(int i=0; i < size-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[size-1]);
    for(int i=0; i < size-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[size-1]);
    for(int i=0; i < size-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[size-1]);
    fclose(fp);
}

void promedioIMG(float *R, float *G, float *B, float *R2, float *G2, float *B2, int size, int L){
    int i, j;
    float sumR, sumG, sumB;
    for(i = 0; i < size; i++){
        sumR = 0; sumG = 0; sumB = 0;
        for(j = 0; j < L; j++){
            sumR += R[i+j*size];
            sumG += G[i+j*size];
            sumB += B[i+j*size];
        }
        R2[i] = sumR/L;
        G2[i] = sumG/L;
        B2[i] = sumB/L;
    }
}

void __global__ kernel(int size, float *R, float *G, float *B, float *Ro, float *Go, float *Bo, int L){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int j;
	if (tid < size){
        float sumR = 0, sumG = 0, sumB = 0;
        for(j = 0; j < L; j++){
            sumR += R[tid+j*size];
            sumG += G[tid+j*size];
            sumB += B[tid+j*size];
        }
        Ro[tid] = sumR/L;
        Go[tid] = sumG/L;
        Bo[tid] = sumB/L;
	}
}

int main(){
    int L, M, N;
    float *R, *G, *B;

    readFile(&L, &M, &N, &R, &G, &B);
    int size = M*N;

    /*
	 *  CPU!!!!!!!!!!
	 */

    float *R2 = (float*)malloc(size*sizeof(float));
    float *G2 = (float*)malloc(size*sizeof(float));
    float *B2 = (float*)malloc(size*sizeof(float));
    clock_t t1,t2;

	t1 = clock();
    promedioIMG(R, G, B, R2, G2, B2, size, L);
    t2 = clock();

    double ms = 1000.0 * (double)(t2-t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f \n", ms);


    char *name = "CPU6.txt";
    writeFile(M, N, R2, G2, B2, name);
    //printf("%f", R2[0]);
    free(R2); free(G2); free(B2);

    /*
	 *  GPU!!!!!!!!!!
	 */

    int bs = 256;
	int gs = (int)ceil(float(size)/bs);

    R2 = (float*)malloc(size*sizeof(float));
    G2 = (float*)malloc(size*sizeof(float));
    B2 = (float*)malloc(size*sizeof(float));

    float *Rdev, *Gdev, *Bdev;
    cudaMalloc(&Rdev, L*size*sizeof(float));
    cudaMalloc(&Gdev, L*size*sizeof(float));
    cudaMalloc(&Bdev, L*size*sizeof(float));

    cudaMemcpy(Rdev, R, L*size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Gdev, G, L*size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Bdev, B, L*size*sizeof(float), cudaMemcpyHostToDevice);


    float *Rdevout, *Gdevout, *Bdevout;
	cudaMalloc(&Rdevout, size*sizeof(float));
	cudaMalloc(&Gdevout, size*sizeof(float));
	cudaMalloc(&Bdevout, size*sizeof(float));

    cudaEvent_t ct1, ct2;
    float dt;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernel<<<gs,bs>>>(size, Rdev, Gdev, Bdev, Rdevout, Gdevout, Bdevout, L);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    printf("Tiempo GPU: %f \n", dt);

    cudaMemcpy(R2, Rdevout, size*sizeof(float), cudaMemcpyDeviceToHost);	
	cudaMemcpy(B2, Bdevout, size*sizeof(float), cudaMemcpyDeviceToHost);	
	cudaMemcpy(G2, Gdevout, size*sizeof(float), cudaMemcpyDeviceToHost);

    writeFile(M, N, R2, G2, B2, "GPU6.txt");	

    free(R2); free(G2); free(B2);
    free(R); free(G); free(B);
}