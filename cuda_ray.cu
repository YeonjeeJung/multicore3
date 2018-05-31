#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define CUDA 0
//#define OPENMP 1
#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048
#define DIVIDENUM 2 

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(Sphere* s, unsigned char* ptr, int idx, int tempDIM)
{

	int xidx = idx / DIVIDENUM;
	int yidx = idx % DIVIDENUM;

        int x = blockIdx.x + xidx*tempDIM;
        int y = blockIdx.y + yidx*tempDIM;
	printf("idx is %d, x:%d y:%d\n",idx,blockIdx.x,blockIdx.y);

	int offset = blockIdx.x + blockIdx.y*tempDIM;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

	//printf("x:%d, y:%d, ox:%f, oy:%f\n",x,y,ox,oy);

	float r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<SPHERES; i++) {
		float   n;
		float   t = s[i].hit( ox, oy, &n );
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		} 
	}

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim,int ydim,int idx, FILE* fp)
{
	int i,x,y;
        if(idx == 0){
		fprintf(fp,"P3\n");
		fprintf(fp,"%d %d\n",DIM, DIM);
		fprintf(fp,"255\n");
	}
	int xStart = (idx / DIVIDENUM) * xdim;
	int xEnd = xStart + xdim;
	int yStart = (idx % DIVIDENUM) * ydim;
	int yEnd = yStart + ydim;

	for (y=yStart;y<yEnd;y++) {
		for (x=xStart;x<xEnd;x++) {
			i=x+y*xdim;
			fprintf(fp,"%d %d %d ",bitmap[4*i],bitmap[4*i+1],bitmap[4*i+2]);
		}
		fprintf(fp,"\n");
	}
}

int main(int argc, char* argv[])
{

	//int no_threads;
	//int option;
	//int x,y;
	//unsigned char* bitmap;

	srand(time(NULL));

	if (argc!=2) {
		printf("> a.out [filename.ppm]\n");
		//printf("[option] 0: CUDA, 1~16: OpenMP using 1~16 threads\n");
		printf("for example, '> a.out result.ppm' means executing CUDA\n");
		exit(0);
	}
        printf("checked arguments\n");

	FILE* fp = fopen(argv[2],"w");

        /*
	if (strcmp(argv[1],"0")==0) option=CUDA;
	else { 
		option=OPENMP;
		no_threads=atoi(argv[1]);
	}
        */

        Sphere temp_s[SPHERES];
        Sphere* ptr_temp_s;
        cudaMalloc( (void**) &ptr_temp_s, sizeof(Sphere)*SPHERES);
	//Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );

        printf("memory allocated in temp_s\n");

        //make __global__ function
	for (int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 2000.0f ) - 1000;
		temp_s[i].y = rnd( 2000.0f ) - 1000;
		temp_s[i].z = rnd( 2000.0f ) - 1000;
		temp_s[i].radius = rnd( 200.0f ) + 40;
	}
	
	int tempDIM = DIM/DIVIDENUM;
        dim3 blocksPerGrid(tempDIM,tempDIM,1);
        //dim3 threadsPerBlock(SPHERES,1,1);
        dim3 threadsPerBlock(1,1,1);

        //unsigned char bitmap[DIM*DIM*4];
        unsigned char bitmap[tempDIM*tempDIM*4];
	unsigned char* ptr_bitmap;
        //cudaMalloc( (void**) &ptr_bitmap, sizeof(unsigned char)*DIM*DIM*4);
        cudaMalloc( (void**) &ptr_bitmap, sizeof(unsigned char)*tempDIM*tempDIM*4);
        //bitmap=(unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM*4);

        printf("memory allocated in bitmap\n");

	int numOfIter = DIVIDENUM*DIVIDENUM;
        for (int i=0; i<numOfIter; i++){
		printf("i is %d\n",i);
        	kernel<<<blocksPerGrid,threadsPerBlock>>>(temp_s,bitmap,i,tempDIM);
        	cudaDeviceSynchronize();
        	/*
		for (x=0;x<DIM;x++) 
			for (y=0;y<DIM;y++) kernel(x,y,temp_s,bitmap);
        	*/
		ppm_write(bitmap,tempDIM,tempDIM,i,fp);

        }
	fclose(fp);
	cudaFree(bitmap);
	cudaFree(temp_s);

	return 0;
}
