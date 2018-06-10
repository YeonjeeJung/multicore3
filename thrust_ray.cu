#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_reference.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

struct Sphere {
	float   r,b,g;
	float   radius;
	float   x,y,z;
	__host__ __device__ float hit( float ox, float oy, float *n ) {
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

struct CalculateBitmap
{
	Sphere* s;
	CalculateBitmap(Sphere* sp) : s(sp) {}
	__host__ __device__ thrust::tuple<unsigned char,unsigned char,unsigned char, unsigned char> operator()(const int& idx)const {
	
		int x = idx / DIM;
		int y = idx % DIM;
	
		//int offset = idx;
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
		
		/*
		ptr[offset*4 + 0] = (int)(r * 255);
		ptr[offset*4 + 1] = (int)(g * 255);
		ptr[offset*4 + 2] = (int)(b * 255);
		ptr[offset*4 + 3] = 255;
		*/
		thrust::tuple<unsigned char, unsigned char, unsigned char, unsigned char> result((int)(r*255),(int)(g*255),(int)(b*255),255);
		return result;
	}
};


void ppm_write(unsigned char* bitmap, int xdim,int ydim, FILE* fp)
{
	int i,x,y;
	fprintf(fp,"P3\n");
	fprintf(fp,"%d %d\n",xdim, ydim);
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
		printf("for example, '> a.out result.ppm' means executing THRUST\n");
		exit(0);
	}
	FILE* fp = fopen(argv[1],"w");


	Sphere* temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
	Sphere* dev_temp_s;
	cudaMalloc( (void**)&dev_temp_s, SPHERES*sizeof(Sphere));

	
	for (int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 2000.0f ) - 1000;
		temp_s[i].y = rnd( 2000.0f ) - 1000;
		temp_s[i].z = rnd( 2000.0f ) - 1000;
		temp_s[i].radius = rnd( 200.0f ) + 40;
	}
	
	cudaMemcpy(dev_temp_s, temp_s, SPHERES*sizeof(Sphere),cudaMemcpyHostToDevice);
	
	
	thrust::device_vector<thrust::tuple<unsigned char, unsigned char, unsigned char, unsigned char> > dev_bitm(DIM*DIM);
	thrust::device_vector<int> idx(DIM*DIM);
	thrust::sequence(idx.begin(),idx.end());

	unsigned char* bitmap = (unsigned char*) malloc(sizeof(unsigned char)*DIM*DIM*4);
	unsigned char* dev_bitmap;
	cudaMalloc((void**)&dev_bitmap,sizeof(unsigned char)*DIM*DIM*4);
	
	clock_t start = clock();
	
	thrust::transform(idx.begin(),idx.end(),dev_bitm.begin(),CalculateBitmap(dev_temp_s));

	clock_t end = clock();

	//printf("end of parallel\n");
	thrust::host_vector<thrust::tuple<unsigned char,unsigned char,unsigned char, unsigned char> > bitm = dev_bitm;

	for(int i=0;i<DIM;i++){
		for(int j=0;j<DIM;j++){
			for(int k=0;k<4;k++){
				bitmap[(i*DIM+j)*4 + 0] = thrust::get<0>(bitm[i+j*DIM]);
				bitmap[(i*DIM+j)*4 + 1] = thrust::get<1>(bitm[i+j*DIM]);
				bitmap[(i*DIM+j)*4 + 2] = thrust::get<2>(bitm[i+j*DIM]);
				bitmap[(i*DIM+j)*4 + 4] = thrust::get<3>(bitm[i+j*DIM]);
			}
		}
	}

	//clock_t end = clock();
	//printf("end of copy\n");
	ppm_write(bitmap,DIM,DIM,fp);

	fclose(fp);
	//free(bitmap);
	//free(temp_s);

	printf("THRUST ray tracing: %1.6f sec\n",(end-start) / (float)CLOCKS_PER_SEC);
	printf("[%s] was generated.\n",argv[1]);

	return 0;
}
