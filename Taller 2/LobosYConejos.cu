#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define MIN 0
#define MAX 4

void readFile(int *M, int *N, int **L, int **C){
    FILE *fp = fopen("initial.txt", "r");
    
    fscanf(fp, "%d %d", M, N);
    int size = (*M)*(*N);
    int i, casilla;
    int *L1 = (int*)malloc(size*sizeof(int));
    int *C1 = (int*)malloc(size*sizeof(int));
    
    for(i = 0; i < size; i++){
        fscanf(fp, "%d", &casilla);
        if(casilla == 1){
            C1[i] = 1;
            L1[i] = 0;
        }
        else if(casilla == 2){
            L1[i] = 1;
            C1[i] = 0;
        }
        else{
            L1[i] = 0;
            C1[i] = 0;
        }
    }
    fclose(fp);
    *L = L1; *C = C1;
}


void writeFile(int M, int N, int *A, char* name){
    FILE *fp = fopen(name, "w");
    int i, j;
    for(i=0; i < M; i++){
        for(j = 0; j < N-1; j++){
            fprintf(fp, "%d ", A[j+i*N]);
        }
        fprintf(fp, "%d\n", A[j+i*N]);
    }
    fclose(fp);
}

void mover(int *L, int *C, int *tempL, int *tempC, int M, int N, int size){
    int i, r;
    srand(432);   // Initialization, should only be called once.
    int contL = 1, contC = 1, x, y; 
    for(i = 0; i < size; i++){
        y = i/N;
        x = i%N; 
        if(L[i] == 1){
            r = rand() % 5;
            //printf("Lobo %d se mueve hacia %d \n", contL, r);
            contL++;
            if(r == 1){ // Derecha
                tempL[(x+1)%N+y*N] += 1;
            }
            else if(r == 2){ // Arriba
                tempL[x+((y-1+M)%M)*N] += 1;
            }
            else if(r == 3){ // Izquierda
                tempL[(x-1+N)%N+y*N] += 1;
            }
            else if(r == 4){ // Abajo
                tempL[x+((y+1)%M)*N] += 1;
            }
            else{
                tempL[i] += 1;
            }
        }
        if(C[i] == 1){
            r = rand() % 5;
            //printf("Conejo %d se mueve hacia %d \n", contC, r);
            contC++;
            if(r == 1){ // Derecha
                tempC[(x+1)%N+y*N] += 1;
            }
            else if(r == 2){ // Arriba
                tempC[x+((y-1+M)%M)*N] += 1;
            }
            else if(r == 3){ // Izquierda
                tempC[(x-1+N)%N+y*N] += 1;
            }
            else if(r == 4){ // Abajo
                tempC[x+((y+1)%M)*N] += 1;
            }
            else{
                tempC[i] += 1; 
            }
        }
        L[i] = 0;
        C[i] = 0;
    }
}

void reproduccion(int *L, int *C, int *tempL, int *tempC, int M, int N, int size){
    int x, y, i;
    for(i = 0; i < size; i++){
        y = i/N;
        x = i%N; 
        if(L[i] >= 2){
            tempL[(x-1+N)%N+y*N] = 1; // Izquierda
            tempL[i] = 1; // Centro
            tempL[(x+1)%N+y*N] = 1; // Derecha
        }
        else if(L[i] == 1){
            tempL[i] = 1; // No se reproduce
        }
        if(C[i] >= 2){
            tempC[x+((y-1+M)%M)*N] = 1; // Arriba
            tempC[i] = 1; // Centro
            tempC[x+((y+1)%M)*N] = 1; // Abajo
        }
        else if(C[i] == 1){
            tempC[i] = 1; // No se reproduce
        }
        L[i] = 0;
        C[i] = 0;
    }
}

void depredacion(int *L, int *C, int *tempL, int *tempC, int size){
    int i;
    for(i = 0; i < size; i++){
        if(L[i] == 1 && C[i] == 1){
            tempC[i] = 0;
        }
        else{
            tempC[i] = C[i];
        }
        tempL[i] = L[i];
        L[i] = 0;
        C[i] = 0;
    }
}

void turno(int *L, int *C, int *tempL, int *tempC, int M, int N, int size){

    mover(L, C, tempL, tempC, M, N, size);

    reproduccion(tempL, tempC, L, C, M, N, size);

    depredacion(L, C, tempL, tempC, size);
}

void contar(int *L, int *C, int *contL, int *contC, int size){
    int i;
    for(i = 0; i < size; i++){
        if(L[i] == 1){
            (*contL)++;
        }
        if(C[i] == 1){
            (*contC)++;
        }
    }
}



__global__ void random_init(curandState *state, int size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < size){
        curand_init(0, tid, 0, &(state[tid]));
    }
    
}


__global__ void cudaMover(int *L, int *C, int *tempL, int *tempC, int M, int N, int size, curandState *state){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < size){
        int r;
        int x, y; 
        y = tid/N;
        x = tid%N; 
        if(L[tid] == 1){
            r = (int)((curand_uniform(&state[tid])-1e-14) * (MAX-MIN+1));
            if(r == 1){ // Derecha
                atomicAdd(&tempL[(x+1)%N+y*N], 1);
            }
            else if(r == 2){ // Arriba
                atomicAdd(&tempL[x+((y-1+M)%M)*N], 1);
            }
            else if(r == 3){ // Izquierda
                atomicAdd(&tempL[(x-1+N)%N+y*N], 1);
            }
            else if(r == 4){ // Abajo
                atomicAdd(&tempL[x+((y+1)%M)*N], 1);
            }
            else{
                atomicAdd(&tempL[tid], 1);
            }
        }
        if(C[tid] == 1){
            r = (int)((curand_uniform(&state[tid])-1e-14) * (MAX-MIN+1));
            if(r == 1){ // Derecha
                atomicAdd(&tempC[(x+1)%N+y*N], 1);
            }
            else if(r == 2){ // Arriba
                atomicAdd(&tempC[x+((y-1+M)%M)*N], 1);
            }
            else if(r == 3){ // Izquierda
                atomicAdd(&tempC[(x-1+N)%N+y*N], 1);
            }
            else if(r == 4){ // Abajo
                atomicAdd(&tempC[x+((y+1)%M)*N], 1);
            }
            else{
                atomicAdd(&tempC[tid], 1); 
            }
        }
        L[tid] = 0;
        C[tid] = 0;
    }
}

__global__ void cudaReproduccion(int *L, int *C, int *tempL, int *tempC, int M, int N, int size){
    int x, y;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < size){
        y = tid/N;
        x = tid%N; 
        if(L[tid] >= 2){
            tempL[(x-1+N)%N+y*N] = 1; // Izquierda OJO AQUI
            tempL[tid] = 1; // Centro
            tempL[(x+1)%N+y*N] = 1; // Derecha
        }
        else if(L[tid] == 1){
            tempL[tid] = 1; // No se reproduce
        }
        if(C[tid] >= 2){
            tempC[x+((y-1+M)%M)*N] = 1; // Arriba
            tempC[tid] = 1; // Centro
            tempC[x+((y+1)%M)*N] = 1; // Abajo
        }
        else if(C[tid] == 1){
            tempC[tid] = 1; // No se reproduce
        }
        L[tid] = 0;
        C[tid] = 0;
    }
}

__global__ void cudaDepredacion(int *L, int *C, int *tempL, int *tempC, int size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < size){
        if(L[tid] == 1 && C[tid] == 1){
            tempC[tid] = 0;
        }
        else{
            tempC[tid] = C[tid];
        }
        tempL[tid] = L[tid];
        L[tid] = 0;
        C[tid] = 0;
    }
}

__global__ void atomicContar(int *L, int *C, int *contL, int *contC, int size){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < size){
        atomicAdd(contL, L[tid]);
        atomicAdd(contC, C[tid]);
    }
}

__global__ void sharedContar(int *L, int size){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int LS[];
    LS[threadIdx.x] = 0;
    if (tId < size){
        LS[threadIdx.x] = L[tId];
        __syncthreads();
        for (int threads = blockDim.x/2; threads > 0; threads /= 2) {
            if (threadIdx.x < threads)
                LS[threadIdx.x] += LS[threadIdx.x + threads];
            __syncthreads();
        }
        if (threadIdx.x == 0){
            L[blockIdx.x] = LS[0];
        }
    }
}


void cudaTurno(int *L, int *C, int *tempL, int *tempC, int M, int N, int size, curandState *states, int gs, int bs){
    cudaMover<<<gs,bs>>>(L, C, tempL, tempC, M, N, size, states);
    cudaReproduccion<<<gs,bs>>>(tempL, tempC, L, C, M, N, size);
    cudaDepredacion<<<gs,bs>>>(L, C, tempL, tempC, size);
}

int main(){
    int M, N, *L, *C, *tempL, *tempC, t1, t2;
    readFile(&M, &N, &L, &C);
    int size = M*N;
    tempL = (int*)calloc(size, sizeof(int));
    tempC = (int*)calloc(size, sizeof(int));    
    int i = 0, contL = 0, contC = 0; 
    t1 = clock();
    while(i < 500){
        //printf("%d\n",i);
        //mover(L, C, tempL, tempC, M, N, size);
        //reproduccion(tempL, tempC, L, C, M, N, size);
        //depredacion(L, C, tempL, tempC, size);
        turno(L, C, tempL, tempC, M, N, size);
        //printf("%d\n",i);
        //mover(tempL, tempC, L, C, M, N, size);
        //reproduccion(L, C, tempL, tempC, M, N, size);
        //depredacion(tempL, tempC, L, C, size);
        turno(tempL, tempC, L, C, M, N, size);
        i++;
    }
    t2 = clock();
    double ms = 1000.0 * (double)(t2-t1) / CLOCKS_PER_SEC;

	printf("Tiempo de simulacion CPU: %f \n", ms);


    t1 = clock();
    contar(L, C, &contL, &contC, size);
    t2 = clock();
    ms = 1000.0 * (double)(t2-t1) / CLOCKS_PER_SEC;

	printf("Tiempo de conteo CPU: %f \n", ms);

    printf("Cantidad de lobos: %d\n", contL);
    printf("Cantidad de conejos: %d\n", contC);
    writeFile(M, N, L, "lobosF.txt");
    writeFile(M, N, C, "conejosF.txt");
    
    free(L); free(C); free(tempL); free(tempC); // XDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
    
	 /*  GPU!!!!!!!!!!
	*/
    int *tempLdev, *tempCdev, *cantL, *cantC;
    readFile(&M, &N, &L, &C);
    //int size = M*N;
    tempL = (int*)calloc(size, sizeof(int));
    tempC = (int*)calloc(size, sizeof(int));    
    int bs = 256;
	int gs = (int)ceil(float(size)/bs);
    int *Ldev, *Cdev;
    cudaMalloc(&Ldev, size*sizeof(int));
    cudaMalloc(&Cdev, size*sizeof(int));

    cudaMemcpy(Ldev, L, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Cdev, C, size*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&tempLdev, size*sizeof(int));
    cudaMalloc(&tempCdev, size*sizeof(int));

    cudaMemcpy(tempLdev, tempL, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(tempCdev, tempC, size*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&cantL, sizeof(int));
    cudaMalloc(&cantC, sizeof(int));

    curandState *states;
    cudaMalloc(&states, size*sizeof(curandState));
    random_init<<<gs,bs>>>(states, size);
    cudaDeviceSynchronize();


    int xd1 = 0;

    cudaMemcpy(cantL, &xd1, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cantC, &xd1, sizeof(int), cudaMemcpyHostToDevice);

    atomicContar<<<gs, bs>>>(Ldev, Cdev, cantL, cantC, size);
    cudaDeviceSynchronize();

    int xd111, xd222;
    cudaMemcpy(&xd111, cantL, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&xd222, cantC, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cantidad de lobos inicial: %d\n", xd111);
    printf("Cantidad de conejos inicial: %d\n", xd222);

    
    

    /*/ ---------------------------- 3 turnos ------------------------------------
    printf("Turno 1\n");
    writeFile(M, N, L, "lobos0.txt");
    printf("XDDD\n");
    writeFile(M, N, C, "conejos0.txt");
    printf("Antes de mover 1\n");
    cudaMover<<<gs, bs>>>(Ldev, Cdev, tempLdev, tempCdev, M, N, size, states);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos1M.txt");
    writeFile(M, N, tempC, "conejos1M.txt");
    printf("Antes de reproducir 1\n");
    cudaReproduccion<<<gs, bs>>>(tempLdev, tempCdev, Ldev, Cdev, M, N, size);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos1R.txt");
    writeFile(M, N, tempC, "conejos1R.txt");
    printf("Antes de depredacion 1\n");
    cudaDepredacion<<<gs, bs>>>(Ldev, Cdev, tempLdev, tempCdev, size);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos1F.txt");
    writeFile(M, N, tempC, "conejos1F.txt");
    printf("\n");

    printf("Turno 2\n");
    cudaMover<<<gs, bs>>>(tempLdev, tempCdev, Ldev, Cdev, M, N, size, states);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos2M.txt");
    writeFile(M, N, tempC, "conejos2M.txt");
    cudaReproduccion<<<gs, bs>>>(Ldev, Cdev, tempLdev, tempCdev, M, N, size);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos2R.txt");
    writeFile(M, N, tempC, "conejos2R.txt");
    cudaDepredacion<<<gs, bs>>>(tempLdev, tempCdev, Ldev, Cdev, size);
    cudaDeviceSynchronize();
    cudaMemcpy(L, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(C, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, L, "lobos2F.txt");
    writeFile(M, N, C, "conejos2F.txt");
    printf("\n");

    printf("Turno 3\n");
    cudaMover<<<gs, bs>>>(Ldev, Cdev, tempLdev, tempCdev, M, N, size, states);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos3M.txt");
    writeFile(M, N, tempC, "conejos3M.txt");
    cudaReproduccion<<<gs, bs>>>(tempLdev, tempCdev, Ldev, Cdev, M, N, size);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos3R.txt");
    writeFile(M, N, tempC, "conejos3R.txt");
    cudaDepredacion<<<gs, bs>>>(Ldev, Cdev, tempLdev, tempCdev, size);
    cudaDeviceSynchronize();
    cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobos3F.txt");
    writeFile(M, N, tempC, "conejos3F.txt");
    printf("\n");

    int xd7 = 0;
    cudaMemcpy(cantL, &xd7, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cantC, &xd7, sizeof(int), cudaMemcpyHostToDevice);
    atomicContar<<<gs, bs>>>(tempLdev, tempCdev, cantL, cantC, size);
    cudaDeviceSynchronize();

    int xd11, xd22;
    cudaMemcpy(&xd11, cantL, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&xd22, cantC, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cantidad de lobos atomic: %d\n", xd11);
    printf("Cantidad de conejos atomic: %d\n", xd22);

    while(gs > 1){
        sharedContar<<<gs, bs, sizeof(int)*bs>>>(tempLdev, size);
        sharedContar<<<gs, bs, sizeof(int)*bs>>>(tempCdev, size);
        gs = int(ceil(float(gs)/bs));
        size = int(ceil(float(size)/bs));
    }
    sharedContar<<<gs, bs, sizeof(int)*bs>>>(tempLdev, size);
    sharedContar<<<gs, bs, sizeof(int)*bs>>>(tempCdev, size);
    cudaDeviceSynchronize();
    int xd123, xd1234;
    cudaMemcpy(&xd123, &(tempLdev[0]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&xd1234, &(tempCdev[0]), sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cantidad de lobos shared: %d\n", xd123);
    printf("Cantidad de conejos shared: %d\n", xd1234);
    

    // -------------------------- 1000 iteraciones -------------------*/
    cudaEvent_t ct1, ct2;
    float dt;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
 
    
    i = 0;
    while(i < 500){
        //char nombre1[10], nombre2[12];
        cudaMover<<<gs,bs>>>(Ldev, Cdev, tempLdev, tempCdev, M, N, size, states);
        cudaDeviceSynchronize();
        /*
        sprintf(nombre1, "lobos%dM.txt", i);
        sprintf(nombre2, "conejos%dM.txt", i);
        cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);*/

        cudaReproduccion<<<gs,bs>>>(tempLdev, tempCdev, Ldev, Cdev, M, N, size);
        cudaDeviceSynchronize();
        /*
        sprintf(nombre1, "lobos%dR.txt", i);
        sprintf(nombre2, "conejos%dR.txt", i);
        cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);
        */
        cudaDepredacion<<<gs,bs>>>(Ldev, Cdev, tempLdev, tempCdev, size);
        cudaDeviceSynchronize();
        
        /*sprintf(nombre1, "lobos%dF.txt", i);
        sprintf(nombre2, "conejos%dF.txt", i);
        cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);*/
        cudaMover<<<gs,bs>>>(tempLdev, tempCdev, Ldev, Cdev, M, N, size, states);
        cudaDeviceSynchronize();
        /*
        sprintf(nombre1, "lobos%dM.txt", i);
        sprintf(nombre2, "conejos%dM.txt", i);
        cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);*/

        cudaReproduccion<<<gs,bs>>>(Ldev, Cdev, tempLdev, tempCdev, M, N, size);
        cudaDeviceSynchronize();

        /*
        sprintf(nombre1, "lobos%dR.txt", i);
        sprintf(nombre2, "conejos%dR.txt", i);
        cudaMemcpy(tempL, tempLdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, tempCdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);*/

        cudaDepredacion<<<gs,bs>>>(tempLdev, tempCdev, Ldev, Cdev, size);
        cudaDeviceSynchronize();

        /*
        sprintf(nombre1, "lobos%dF.txt", i);
        sprintf(nombre2, "conejos%dF.txt", i);
        cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
        writeFile(M, N, tempL, nombre1);
        writeFile(M, N, tempC, nombre2);*/
        /*
        cudaTurno(Ldev, Cdev, tempLdev, tempCdev, M, N, size, states, gs, bs);
        cudaTurno(tempLdev, tempCdev, Ldev, Cdev, M, N, size, states, gs, bs);*/
        i++;
    }
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    printf("Tiempo GPU: %f \n", dt);

    cudaMemcpy(tempL, Ldev, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tempC, Cdev, size*sizeof(int), cudaMemcpyDeviceToHost);
    writeFile(M, N, tempL, "lobosF.txt");
    writeFile(M, N, tempC, "conejosF.txt");
    printf("\n");

    int xd7 = 0;
    cudaMemcpy(cantL, &xd7, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cantC, &xd7, sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t cta1, cta2;
    float dta;
    cudaEventCreate(&cta1);
    cudaEventCreate(&cta2);
    cudaEventRecord(cta1);
    atomicContar<<<gs, bs>>>(Ldev, Cdev, cantL, cantC, size);
    cudaDeviceSynchronize();
    cudaEventRecord(cta2);
    cudaEventSynchronize(cta2);
    cudaEventElapsedTime(&dta, cta1, cta2);
    printf("Tiempo GPU atomic: %f \n", dta);

    int xd11, xd22;
    cudaMemcpy(&xd11, cantL, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&xd22, cantC, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cantidad de lobos atomic: %d\n", xd11);
    printf("Cantidad de conejos atomic: %d\n", xd22);

    
    cudaEvent_t cts1, cts2;
    float dts;
    cudaEventCreate(&cts1);
    cudaEventCreate(&cts2);
    cudaEventRecord(cts1);

    while(gs > 1){
        sharedContar<<<gs, bs, sizeof(int)*bs>>>(Ldev, size);
        sharedContar<<<gs, bs, sizeof(int)*bs>>>(Cdev, size);
        gs = int(ceil(float(gs)/bs));
        size = int(ceil(float(size)/bs));
    }
    sharedContar<<<gs, bs, sizeof(int)*bs>>>(Ldev, size);
    sharedContar<<<gs, bs, sizeof(int)*bs>>>(Cdev, size);
    cudaDeviceSynchronize();

    cudaEventRecord(cts2);
    cudaEventSynchronize(cts2);
    cudaEventElapsedTime(&dts, cts1, cts2);
    printf("Tiempo GPU shared: %f \n", dts);

    int xd123, xd1234;
    cudaMemcpy(&xd123, &(Ldev[0]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&xd1234, &(Cdev[0]), sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cantidad de lobos shared: %d\n", xd123);
    printf("Cantidad de conejos shared: %d\n", xd1234);

    // ---------------------------- Tiempos -------------------------------
    /*
    cudaEvent_t ct1, ct2;
    float dt;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    i = 0;
    while(i < 500){
        cudaTurno(Ldev, Cdev, tempLdev, tempCdev, M, N, size, states, gs, bs);
        cudaTurno(tempLdev, tempCdev, Ldev, Cdev, M, N, size, states, gs, bs);
        i++;
    }
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    printf("Tiempo GPU: %f \n", dt);

    atomicContar<<<gs,bs>>>(Ldev, Cdev, cantL, cantC, size);
    */

    //writeFile(M, N, R2, G2, B2, "GPU6.txt");	

    free(L); free(C);
    free(tempL); free(tempC); // free GPU-??????   
    
    /*
    printf("Turno 1\n");
    writeFile(M, N, L, "lobos0.txt");
    writeFile(M, N, C, "conejos0.txt");
    mover(L, C, tempL, tempC, M, N, size);
    writeFile(M, N, tempL, "lobos1M.txt");
    writeFile(M, N, tempC, "conejos1M.txt");
    reproduccion(tempL, tempC, L, C, M, N, size);
    writeFile(M, N, L, "lobos1R.txt");
    writeFile(M, N, C, "conejos1R.txt");
    depredacion(L, C, tempL, tempC, size);
    writeFile(M, N, tempL, "lobos1F.txt");
    writeFile(M, N, tempC, "conejos1F.txt");
    printf("\n");

    printf("Turno 2\n");
    mover(tempL, tempC, L, C, M, N, size);
    writeFile(M, N, L, "lobos2M.txt");
    writeFile(M, N, C, "conejos2M.txt");
    reproduccion(L, C, tempL, tempC, M, N, size);
    writeFile(M, N, tempL, "lobos2R.txt");
    writeFile(M, N, tempC, "conejos2R.txt");
    depredacion(tempL, tempC, L, C, size);
    writeFile(M, N, L, "lobos2F.txt");
    writeFile(M, N, C, "conejos2F.txt");
    printf("\n");

    printf("Turno 3\n");
    mover(L, C, tempL, tempC, M, N, size);
    writeFile(M, N, tempL, "lobos3M.txt");
    writeFile(M, N, tempC, "conejos3M.txt");
    reproduccion(tempL, tempC, L, C, M, N, size);
    writeFile(M, N, L, "lobos3R.txt");
    writeFile(M, N, C, "conejos3R.txt");
    depredacion(L, C, tempL, tempC, size);
    writeFile(M, N, tempL, "lobos3F.txt");
    writeFile(M, N, tempC, "conejos3F.txt");
    printf("\n");
    contar(tempL, tempC, &contL, &contC, size);*/

    return 0;
}