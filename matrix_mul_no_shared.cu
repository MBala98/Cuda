#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "assert.h"
#define BLOCK_SIZE 32
__global__ void MatrixMul(int *A, int *B, int *C, int N) {
    /* One thread for one result matrix element.
     * block_row and block_col create sub matrix in result matrix C, that goes from
     * C[SUB_SIZE*block_row][SUB_SIZE*block_col] to C[SUB_SIZE*(block_row+1)-1][SUB_SIZE*(block_col+1)-1]*/
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    //  printf("%d %d %d %d %d %d\n", block_row, block_col, row, col, blockDim.y, blockDim.x);
 /*   int result = 0;
    int number_of_sub_matrixes = (N + SUB_SIZE -1)/SUB_SIZE;
    // Iterating through every sub matrix needed to calculate one field
    for (int m = 0; m <  number_of_sub_matrixes; m++) {
        __shared__ int A_sub[SUB_SIZE][SUB_SIZE];
        __shared__ int B_sub[SUB_SIZE][SUB_SIZE];
        if ((SUB_SIZE*m + col) < N) {
            A_sub[row][col] = A[SUB_SIZE * (block_row) * N + row * N + SUB_SIZE * m + col];
        }
        else {
            A_sub[row][col] = 0;
        }
        if ((m*SUB_SIZE + row) < N) {
            B_sub[row][col] = B[m * SUB_SIZE * N + row * N + block_col * SUB_SIZE + col];
        }
        else {
            B_sub[row][col] = 0;
        }
        __syncthreads();
        // iterating through row and column in submatrixes to calculate field
        for (int j = 0; j < SUB_SIZE; j++) {
            result += A_sub[row][j] * B_sub[j][col];
        }
        __syncthreads();
    }
    if (((SUB_SIZE*block_row+row) < N) && (SUB_SIZE*block_col+col < N)) {
        //printf("saving result with %d %d %d %d into %d %d\n", block_row, block_col, row, col, SUB_SIZE*block_row*N +row*N + SUB_SIZE*block_col+col, result);
        C[SUB_SIZE*block_row*N + row*N + SUB_SIZE*block_col+col] = result;
    }
    else {
        //printf("dispatched with %d %d %d %d %d %d\n", block_row, block_col, row, col, SUB_SIZE*block_row*N +row*N + SUB_SIZE*block_col+col, result);
    }
    */
    for (int i = 0; i < N; i++) {
        if (((BLOCK_SIZE*block_row+row) < N) && (BLOCK_SIZE*block_col+col < N)) {
            C[BLOCK_SIZE * block_row * N + row * N + BLOCK_SIZE * block_col + col] += A[BLOCK_SIZE * block_row * N + row * N + i]*B[i*N + BLOCK_SIZE * block_col + col];
        }
    }

}






using namespace std;
int main(int argc, char **argv) {
    assert(argc > 1);
    int liczba = atoi(argv[1]);
    int *tablica_cpu = (int*)malloc(sizeof(int)*liczba*liczba);
    int *tablica_cpu2 = (int*)malloc(sizeof(int)*liczba*liczba);
    int *tablica_cpu3 = (int*)malloc(sizeof(int)*liczba*liczba);
    for (int i = 0; i < liczba; i++) {
        for (int j = 0; j < liczba; j++) {
            tablica_cpu[i*liczba + j] = 2;
            tablica_cpu2[i*liczba + j] = 2;
            tablica_cpu3[i*liczba + j] = 0;
        }
    }
    int *tablica_gpu;
    int *tablica_gpu2;
    int *tablica_gpu3;
    cudaError_t status;
    status = cudaMalloc(&tablica_gpu, sizeof(int) * liczba*liczba);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaMalloc(&tablica_gpu2, sizeof(int) * liczba*liczba);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaMalloc(&tablica_gpu3, sizeof(int) * liczba*liczba);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }

    status = cudaMemcpy(tablica_gpu, tablica_cpu, sizeof(int) * liczba*liczba, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaMemcpy(tablica_gpu2, tablica_cpu2, sizeof(int) * liczba*liczba, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaMemcpy(tablica_gpu3, tablica_cpu3, sizeof(int) * liczba*liczba, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //printf("%d %d \n", dimBlock.x, dimBlock.y);
    dim3 dimGrid((liczba + dimBlock.x -1)/dimBlock.x, (liczba + dimBlock.y -1)/dimBlock.y);
   // printf("%d %d\n", dimGrid.x, dimGrid.y);
    MatrixMul<<<dimGrid, dimBlock>>>(tablica_gpu, tablica_gpu2, tablica_gpu3, liczba);
    cudaDeviceSynchronize();
    status = cudaMemcpy(tablica_cpu3, tablica_gpu3, sizeof(int) * liczba*liczba, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    for (int i = 0; i < liczba; i++) {
        for (int j = 0; j < liczba; j++) {
          //     cout << tablica_cpu3[i*liczba + j] << '\n';
        }
    }

    status = cudaFree(tablica_gpu);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaFree(tablica_gpu2);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    status = cudaFree(tablica_gpu3);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << '\n';
    }
    free(tablica_cpu);
    free(tablica_cpu2);
    free(tablica_cpu3);

}