#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__device__ unsigned long long int gcd(unsigned long long int a, unsigned long long int b){
	unsigned long long int r=0;
	while(b!=0){
		r = a%b;
		a = b;
		b = r;
	}
	return a;
}

__global__ void MonteCarlo(unsigned long long int n, unsigned long long int *d){
	unsigned long long int dtmp = 1;
	unsigned long long int a = threadIdx.x;
	unsigned long long int b = threadIdx.x;
	while((dtmp==1||dtmp==n) && (*d==1||*d==n)){
		a = a*a+a+1;
		b = b*b+b+1;
		b = b*b+b+1;
		dtmp = gcd(a-b,n);
	}
	*d=dtmp;
}

int main(int argc, char *argv[]){
	if(argc<2)
		exit(0);
	unsigned long long int n = atoll(argv[1]);
	unsigned long long int *ptrd;
	unsigned long long int d = 1;
	cudaMalloc((void**) &ptrd, sizeof(unsigned long long int));
	cudaMemcpy(ptrd, &d, sizeof(unsigned long long int),cudaMemcpyHostToDevice);

	MonteCarlo<<<1,5>>>(n,ptrd);

	cudaMemcpy(&d, ptrd, sizeof(unsigned long long int),cudaMemcpyDeviceToHost);

	printf("%lld\n",d);
	cudaFree(ptrd);
	return 0;
}
