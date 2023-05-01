#include <stdio.h>

/*
 *	 DESCRIPTION
 *		CUDA Kernel demonstrating Gaussian Blur via moving window over an Image
 *		
 *			1  2  5  2  0  3
 *			   -------
 *			3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
 *			  |       |
 *			4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
 *			  |       |
 *			0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
 *			   -------
 *			9  6  5  0  3  9
 *	 
 *	 TODO
 *		General code polishing.
 *		
 */
	
#include "utils.h"  // Email me for more info.

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{ 
	int absolute_image_position_x = blockIdx.x;
	int absolute_image_position_y = threadIdx.x;
	int idx =  numRows*absolute_image_position_x + absolute_image_position_y; 
   
             
	if (absolute_image_position_x >= numCols ||
		absolute_image_position_y >= numRows ) {
		return;
	}
		
	int filter_len = filterWidth * filterWidth;
	int filter_rad = filterWidth / 2;
	int stencil_row = 0, stencil_col = 0, idx_clamped = 0;
	float blur_value = 0;
	
	for (int idx_filter = 0; idx_filter < filter_len; idx_filter++) {
		stencil_row = idx_filter / filter_len;
		stencil_col = idx_filter % filter_len; 
		idx_clamped = idx - filter_rad * (numCols + 1) + stencil_row * numCols + stencil_col;
		if (idx_clamped < 0) idx_clamped = 0;
		if (idx_clamped > numRows * numCols - 1) idx_clamped = numRows * numCols - 1;
		blur_value += inputChannel[idx_clamped] * filter[idx_filter];
	}
	__syncthreads();
	
	outputChannel[idx] = blur_value;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int absolute_image_position_x = blockIdx.x;
  int absolute_image_position_y = threadIdx.x;
  int idx =  numRows*absolute_image_position_x + absolute_image_position_y; 
  
  if (  absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows ) {
		return;
  }
   
  redChannel[idx]   = inputImageRGBA[idx].x;
  greenChannel[idx] = inputImageRGBA[idx].y;
  blueChannel[idx]  = inputImageRGBA[idx].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // Careful not to access memory that is outside of the image
  // with threads that are mapped there return too early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  // Alpha -> 255 for non-transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMemset(d_red,   0, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMemset(d_green,   0, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMemset(d_blue,   0, sizeof(unsigned char) * numRowsImage * numColsImage));

  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

  // Copy filter from Host to GPU
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(numRows,1,1);
  const dim3 gridSize(numCols,numRows,1);

  // Launch
  separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  // Do the error management
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Convolution call: red channel
  gaussian_blur<<<gridSize,blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth); 
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
  // ...: green channel
  gaussian_blur<<<gridSize,blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
  // ...: blue channel
  gaussian_blur<<<gridSize,blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // Recombine the results.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
