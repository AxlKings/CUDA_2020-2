#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

void ReadFile(float **L, int *M, int *N, int T){
    FILE *fp = fopen("initial80x160.txt", "r");

    fscanf(fp, "%d %d", M, N);

    int size = (*M)*(*N);

    int i;
    //printf("voy a hacer el malloc\n");
    float *L1 = (float*)malloc(size*T*sizeof(float));
    //printf("hice el malloc\n");

    for(i = 0; i < size; i++){
        fscanf(fp, "%f", &L1[i]); 
        //printf("%f", L1[i]);
    }

    fclose(fp);
    *L = L1;
}

void WriteFile(int M, int N, int T, float *L, char* name){
    FILE *fp = fopen(name, "w");

    fprintf(fp, "%d %d %d\n", M, N, T);

    int size = N*M;

    for(int i=0; i < T*size; i++){
        //printf("%d \n", i+j*N);
        fprintf(fp, "%f ", L[i]);
    }
    fprintf(fp, "\n");
        
    fclose(fp);
}


void CPU_Calor(int M, int N, float *U, int t){
    int Dx = 1, Dy = 1, Dt = 1, x, y, up, down, left, right, center;
    float alpha = 0.1;

    int size = M*N;

    for (int i = 0; i < size; i++){
        y = i/N;
        x = i%N;
        //printf("x: %d  y: %d \n",x,y);

        center = U[(x+y*N)+((t-1)*size)];
        if(x == 0){
            left = 0;
            right = U[((x+1)+y*N)+((t-1)*size)];
        }
        else if(x == N-1){
            left = U[((x-1)+y*N)+((t-1)*size)];
            right = 0;
        }
        else{
            left = U[((x-1)+y*N)+((t-1)*size)];
            right = U[((x+1)+y*N)+((t-1)*size)];
        }
        if(y == 0){
            up = 0;
            down = U[(x+(y+1)*N)+((t-1)*size)];
            
        }
        else if(y == M-1){
            up = U[(x+(y-1)*N)+((t-1)*size)];
            down = 0;
        }
        else{
            down = U[(x+(y+1)*N)+((t-1)*size)];
            up = U[(x+(y-1)*N)+((t-1)*size)];
        }
        U[(x+y*N)+t*size] = center + Dt * alpha *( ( ( left - 2 * center + right ) / Dx ) + ( (down - 2 * center + up ) / Dy ) );
        
    }
    
}

__global__ void GPU_Calor(int M, int N, float *U, int t){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int Dx = 1, Dy = 1, Dt = 1, x, y, up, down, left, right, center;
    float alpha = 0.1;

    int size = M*N;

    if(tId < size){
        y = tId/N;
        x = tId%N;
        //printf("x: %d  y: %d \n",x,y);

        center = U[(x+y*N)+((t-1)*size)];
        if(x == 0){
            left = 0;
            right = U[((x+1)+y*N)+((t-1)*size)];
        }
        else if(x == N-1){
            left = U[((x-1)+y*N)+((t-1)*size)];
            right = 0;
        }
        else{
            left = U[((x-1)+y*N)+((t-1)*size)];
            right = U[((x+1)+y*N)+((t-1)*size)];
        }
        if(y == 0){
            up = 0;
            down = U[(x+(y+1)*N)+((t-1)*size)];
            
        }
        else if(y == M-1){
            up = U[(x+(y-1)*N)+((t-1)*size)];
            down = 0;
        }
        else{
            down = U[(x+(y+1)*N)+((t-1)*size)];
            up = U[(x+(y-1)*N)+((t-1)*size)];
        }
        U[(x+y*N)+t*size] = center + Dt * alpha *( ( ( left - 2 * center + right ) / Dx ) + ( (down - 2 * center + up ) / Dy ) );
        
    }
    
}

__global__ void Shared_Calor(int M, int N, float *U, int t){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int Dx = 1, Dy = 1, Dt = 1, x, y, up, down, left, right, center, X, Y;
    float alpha = 0.1;

    int size = M*N;

    if(tId < size){
        x = threadIdx.x%16;
        y = threadIdx.x/16;
        X = blockIdx.x%(N/16);
        Y = blockIdx.x/(N/16);


        extern __shared__ float LS[];
        
        //printf("x: %d  y: %d \n",x,y);

        center = U[(X*16)+(Y*16*N)+(x)+(y*N)+(t-1)*size];
        LS[threadIdx.x] = center;
        __syncthreads();
        if(x == 0 && X == 0){
            left = 0;
            right = LS[(x+1)+y*16];
        }
        else if(x == 15 && X == (N/16)-1){
            left = LS[(x-1)+y*16];
            right = 0;
        }
        else if(x == 0){
            left = U[((X*16)+(Y*16*N)+(x-1)+(y*N))+(t-1)*size];
            right = LS[(x+1)+y*16];

        }
        else if(x == 15){
            left = LS[(x-1)+y*16];
            right = U[((X*16)+(Y*16*N)+(x+1)+(y*N))+(t-1)*size];
        }   
        else{
            left = LS[(x-1)+y*16];
            right = LS[(x+1)+y*16];
        } // XDDDDDDDDDDDD
        if(y == 0 && Y == 0){
            up = 0;
            down = LS[x+(y+1)*16];
        }
        else if(y == 15 && Y == (M/16)-1){
            up = LS[x+(y-1)*16];
            down = 0;
        }
        else if(y == 0){
            up = U[((X*16)+(Y*16*N)+(x)+((y-1)*N))+(t-1)*size];
            down = LS[x+(y+1)*16];
        }
        else if(y == 15){
            up = LS[x+(y-1)*16];
            down = U[((X*16)+(Y*16*N)+(x)+((y+1)*N))+(t-1)*size];
        }
        else{
            up = LS[x+(y-1)*16];
            down = LS[x+(y+1)*16];
        }
        U[(X*16)+(Y*16*N)+(x)+(y*N)+t*size] = center + Dt * alpha *( ( ( left - 2 * center + right ) / Dx ) + ( (down - 2 * center + up ) / Dy ) );
        
    }
    
}

int main(int argc, char const *argv[]){
    int M, N, T, t;
    float *U;
    clock_t t1,t2;
    
    // ---------------------- CPU ------------------------------
    //WriteFile(M,N, U, "Salida.txt");
    printf("Ingrese cantidad de iteraciones: ");
    scanf("%d",&T);
    ReadFile(&U, &M, &N, T);

    //float *Utemp = (float *)malloc(size*sizeof(float));
    t1 = clock();
    for (t = 1; t < T; t++){
        CPU_Calor(M, N, U, t);
    }
    t2 = clock();
    double ms = 1000.0 * (double)(t2-t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f \n", ms);

    WriteFile(M, N, T, U, "SalidaCPU.txt");


    free(U);

    // ---------------------- GPU ------------------------------
    
    ReadFile(&U, &M, &N, T);
    float *Udev, *UdevShared;
    int size = N*M;
    int bs = 256;
	int gs = (int)ceil(float(size)/bs);
    

    cudaMalloc(&Udev, T*size*sizeof(float));
    cudaMemcpy(Udev, U, T*size*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t ct1, ct2;
    float dt, dtt;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    for(t = 1; t < T; t++){
        GPU_Calor<<<gs,bs>>>(M,N,Udev,t);
    }
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    printf("Tiempo GPU: %f \n", dt);

    cudaMemcpy(U, Udev, T*size*sizeof(float), cudaMemcpyDeviceToHost);
    WriteFile(M, N, T, U, "SalidaGPU.txt");
    free(U);

    // ------------------ GPU Shared --------------------------
    ReadFile(&U, &M, &N, T);
    cudaMalloc(&UdevShared, T*size*sizeof(float));
    cudaMemcpy(UdevShared, U, T*size*sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t ct11, ct22;
    cudaEventCreate(&ct11);
    cudaEventCreate(&ct22);
    cudaEventRecord(ct11);
    for(t = 1; t < T; t++){
        Shared_Calor<<<gs,bs, sizeof(float)*bs>>>(M,N,UdevShared,t);
    }
    cudaEventRecord(ct22);
    cudaEventSynchronize(ct22);
    cudaEventElapsedTime(&dtt, ct11, ct22);
    printf("Tiempo GPU Shared: %f \n", dtt);

    cudaMemcpy(U, UdevShared, T*size*sizeof(float), cudaMemcpyDeviceToHost);
    WriteFile(M, N, T, U, "SalidaGPU_Shared.txt");
    free(U);
    return 0;
}
