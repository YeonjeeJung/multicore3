#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

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

__global__ void kernel(Sphere* s, unsigned char* ptr)
{

        int x = blockIdx.x;
        int y = blockIdx.y;

	int offset = x + y*DIM;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

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
	//printf("x:%d y:%d\n",x,y);
}

void ppm_write(unsigned char* bitmap, int xdim,int ydim,FILE* fp)
{
	int i,x,y;
	fprintf(fp,"P3\n");
	fprintf(fp,"%d %d\n",DIM, DIM);
	fprintf(fp,"255\n");

	for (y=0;y<ydim;y++) {
		for (x=0;x<xdim;x++) {
			i=x+y*xdim;
			fprintf(fp,"%d %d %d ",bitmap[4*i],bitmap[4*i+1],bitmap[4*i+2]);
		}
		fprintf(fp,"\n");
	}
}

int main(int argc, char* argv[])
{
	srand(time(NULL));

	if (argc!=2) {
		printf("> a.out [filename.ppm]\n");
		printf("for example, '> a.out result.ppm' means executing CUDA\n");
		exit(0);
	}
        //printf("checked arguments\n");
	FILE* fp = fopen(argv[1],"w");
	

	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
	Sphere* dev_temp_s;
	cudaMalloc( (void**)&dev_temp_s, SPHERES*sizeof(Sphere));

        //printf("memory allocated in temp_s\n");

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

	cudaMemcpy( dev_temp_s, temp_s, SPHERES*sizeof(Sphere), cudaMemcpyHostToDevice);

        dim3 blocksPerGrid(DIM,DIM,1);
        dim3 threadsPerBlock(1,1,1);
	//dim3 threadsPerBlock(SPHERES,1,1);

	//dim3 blocksPerGrid(1,1);
	//dim3 threadsPerBlock(DIM,DIM);

	unsigned char* bitmap;
	bitmap=(unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM*4);
	unsigned char* dev_bitmap;
	cudaMalloc( (void**)&dev_bitmap, DIM*DIM*4*sizeof(unsigned char));

        //printf("memory allocated in bitmap\n");
	
	clock_t start = clock();
	kernel<<<blocksPerGrid,threadsPerBlock>>>(dev_temp_s,dev_bitmap);
	cudaMemcpy(bitmap,dev_bitmap,DIM*DIM*4*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	clock_t end = clock();
	ppm_write(bitmap,DIM,DIM,fp);
	fclose(fp);

	cudaFree(dev_bitmap);
	cudaFree(dev_temp_s);

	printf("CUDA ray tracing: %1.2f sec\n", (end-start)/(float)CLOCKS_PER_SEC);
	printf("[%s] was generated.\n",argv[1]);

	return 0;
}
