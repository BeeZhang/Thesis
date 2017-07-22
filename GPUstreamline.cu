//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <stdlib.h>
#include <math.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
//#include <thrust/sequence.h>
#include <cuda_runtime.h>
//#include <cublacs.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStructuredPointsReader.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDataObject.h"
#include "vtkSmartPointer.h"
#include <vtkFieldData.h>
#include <vtkStructuredPoints.h>
#include <vtkDataSetAttributes.h>
#include <vtkImageReader.h>
#include <iostream>
#include "vtkConfigure.h"
#include <vtkStructuredPointsWriter.h>
using namespace std;



__device__ float* PointVelocityInterpolation(float* point,float* InputData,float* InputSpace, int* InputDims);

__global__ void GetStreamLineLastPoints(float* seeds,float* InputData,float* InputSpace,
	int* InputDims,int step,float timestep,float* StreamLineLastPoints);


__device__ float* PointVelocityInterpolation(float* point,float* InputData,float* InputSpace, int* InputDims)
{
	float subData[24];//8 velocities and 3 values for every velocity
	float dx=InputSpace[0];
	float dy=InputSpace[1];
	float dt=InputSpace[2];
    int InPX=floor(point[0]/dx);//x left of the initialpoint
    int InPY=floor(point[1]/dy);// Y below of the initialpoint
    int InPt=floor(point[2]/dt);//the time below the initialpoint
    int xp=InputDims[0];
    int yp=InputDims[1];
    int tp=InputDims[2];
	int PXY=xp*yp; //how many points in one slice
	float PointVelocityFinal[3];
	float PointVelocity0[3];
	float PointVelocity1[3];
	//cout<<"for point ("<<point[0]<<","<<point[1]<<","<<point[2]<<")"<<endl;
	if(InPX<0||InPX>(xp-1)||InPY<0||InPY>(yp-1)||InPt<0||InPt>2)//only three slice
	{
		//cout<<"the point is out of the area now, Streamline ends"<<endl;
		PointVelocityFinal[2]=1;
	}
	else
	{
		for(int i=0;i<8;i++)
		{
			subData[i*3]=InputData[3*(InPt*PXY+InPY*yp+InPX+i)];
			subData[i*3+1]=InputData[3*(InPt*PXY+InPY*yp+InPX+i)+1];
			subData[i*3+2]=InputData[3*(InPt*PXY+InPY*yp+InPX+i)+2];
			//cout<<"subVelocity of "<<i<<" ("<<subData[i*3]<<","<<subData[i*3+1]<<","<<subData[i*3+2]<<")"<<endl;
		}
	    float d0=point[0]-dx*InPX;//the distance from the first point in x axic
	    float d1=point[1]-dy*InPY;//the distance from the first point in y axic
		PointVelocity0[0]=(1-d1/dy)*((1-d0/dx)*subData[0]+(d0/dx)*subData[3])+(d1/dy)*((1-d0/dx)*subData[6]+(d0/dx)*subData[9]);
	    PointVelocity0[1]=(1-d1/dy)*((1-d0/dx)*subData[1]+(d0/dx)*subData[4])+(d1/dy)*((1-d0/dx)*subData[7]+(d0/dx)*subData[10]);
	    PointVelocity1[0]=(1-d1/dy)*((1-d0/dx)*subData[12]+(d0/dx)*subData[15])+(d1/dy)*((1-d0/dx)*subData[18]+(d0/dx)*subData[21]);
	    PointVelocity1[1]=(1-d1/dy)*((1-d0/dx)*subData[13]+(d0/dx)*subData[16])+(d1/dy)*((1-d0/dx)*subData[19]+(d0/dx)*subData[22]);
         //cout<<"InitialVelocity "<<InitialsubData[0]<<" "<<InitialsubData[1]<<endl;
		float d2=point[2]-dt*InPt;
		PointVelocityFinal[0]=(1-d2/dt)*PointVelocity0[0]+(d2/dt)*PointVelocity1[0];
		PointVelocityFinal[1]=(1-d2/dt)*PointVelocity0[1]+(d2/dt)*PointVelocity1[1];
		PointVelocityFinal[2]=0;
		//cout<<"interpolation Velocity of "<<" ("<<PointVelocityFinal[0]<<","<<PointVelocityFinal[1]<<","<<PointVelocityFinal[2]<<")"<<endl;

	}
	return PointVelocityFinal;
}

__global__ void GetStreamLinePoints(float* seeds,float* InputData,float* InputSpace,
	int* InputDims,int step,float timestep,float* StreamLineLastPoints, int xs, int ys)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int N=(xs+1)*(ys+1);
    if(index<N)
    {
    	
    	float OneSeed[3];
    	OneSeed[0]=seeds[index*3];
    	OneSeed[1]=seeds[index*3+1];
    	OneSeed[2]=seeds[index*3+2];
        //cout<<"point "<<index<<"("<<OneSeed[0]<<","<<OneSeed[1]<<","<<OneSeed[2]<<")"<<endl;
    	float* InitialVelocity;
		for (int i=0; i<step;i++)
		{
			InitialVelocity=PointVelocityInterpolation(OneSeed,InputData,InputSpace,InputDims);
			if (InitialVelocity[0]==0&&InitialVelocity[1]==0)
		    {
			    //std::cout<<"seed "<<" at step "<<i<<" ,the velocity is 0, streamline finished"<<endl;
			    break;
		    }
		    if(InitialVelocity[2]==1)
		    	{//std::cout<<"point "<<"at step "<<i<<endl;
		            break;
		        }
		    else
		    {
		        float K1[3];
		        K1[0]=InitialVelocity[0];
		        K1[1]=InitialVelocity[1];
		        K1[2]=InitialVelocity[2];
		        float K2Point[3];
		        K2Point[0]=OneSeed[0]+timestep*K1[0]*0.5;
		        K2Point[1]=OneSeed[1]+timestep*K1[1]*0.5;
		        K2Point[2]=OneSeed[2];
		        float* K2;
		        K2=PointVelocityInterpolation(K2Point,InputData,InputSpace,InputDims);
		        if(K2[2]==1)
		        {//cout<<"point "<<index<<" at step "<<i<<endl;
		            break;
		        }
			    else
			    {
				    float K3Point[3];
		            K3Point[0]=OneSeed[0]+timestep*K2[0]*0.5;
		            K3Point[1]=OneSeed[1]+timestep*K2[1]*0.5;
		            K3Point[2]=OneSeed[2];
		            float* K3=PointVelocityInterpolation(K3Point,InputData,InputSpace,InputDims);;
		           if(K3[2]==1)
		           {//cout<<"point "<<"at step "<<i<<endl;
		                break;
		            }
				   else
				   {
				        float K4Point[3];
		                K4Point[0]=OneSeed[0]+timestep*K3[0];
		                K4Point[1]=OneSeed[1]+timestep*K3[1];
	                    K4Point[2]=OneSeed[2];
		                float* K4;
		                K4=PointVelocityInterpolation(K4Point,InputData,InputSpace,InputDims);;
		                if(K4[2]==1)
		                {//cout<<"point "<<"at step "<<i<<endl;
		                    break;
		                }
		                else
		               {
		                    OneSeed[0]=OneSeed[0]+timestep*(K1[0]+2*K2[0]+2*K3[0]+K4[0])/6;
		                    OneSeed[1]=OneSeed[1]+timestep*(K1[1]+2*K2[1]+2*K3[1]+K4[1])/6;
		                    OneSeed[2]=OneSeed[2];
					   }
		            
		           }
		       }
		   }
		   	   // cout<<"point "<<index<<" "<<i<<"("<<OneSeed[0]<<","<<OneSeed[1]<<","<<OneSeed[2]<<")"<<endl;
	    }

	    StreamLineLastPoints[index*3]=OneSeed[0];
        StreamLineLastPoints[index*3+1]=OneSeed[1];
        StreamLineLastPoints[index*3+2]=OneSeed[2];
    }
    index = blockDim.x * blockIdx.x +index;
}



int main(void)
{
	// Read the file
	vtkSmartPointer<vtkStructuredPointsReader> reader = vtkSmartPointer<vtkStructuredPointsReader>::New();
    //reader->SetFileName("createlessdata.vtk");
    reader->SetFileName("velotimeWeight.vtk");
    //reader->SetFileName("velotimeWeight1.vtk");
    reader->Update();
    vtkSmartPointer<vtkStructuredPoints> Data= vtkSmartPointer<vtkStructuredPoints>::New();
    Data = reader->GetOutput();  
    int* dims = Data->GetDimensions();
    double* space1 = Data->GetSpacing();
    float space[3];
    space[0]=(float)space1[0];
    space[1]=(float)space1[1];
    space[2]=(float)space1[2];
    std::cout<<"here is the space  "<<space[0]<<" "<<space[1]<<" "<<space[2]<<std::endl;
    std::cout<<"here is the dims  "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<std::endl;
    vtkPointData* points;
    points = Data->GetPointData();
    vtkDataArray* OriginalData;
    OriginalData = points->GetArray(0);
    int NumberofTuple=OriginalData->GetNumberOfTuples();
	float startX=17;
	float endX=20;
    float startY=17;
	float endY=20;
	float startT=100;
	float endT=200;
	int intstartX=floor(startX);
	int intendX=ceil(endX);
	int intstartY=floor(startY);
	int intendY=ceil(endY);
	int intstartT=floor(startT);
	int intendT=ceil(endT);		
	int StepsPerCell=1;
	int Slice=1;
	int Ndims[3]={(intendX-intstartX)*StepsPerCell+1,(intendY-intstartY)*StepsPerCell+1,Slice};

    //device_vector<float> d_InputData(dims[0]*dims[1]*3*3);
    float InputData[dims[0]*dims[1]*3*3];
    int intT=intstartT;
    for(int i=0;i<3*dims[0]*dims[1];i++)//for streamline three layer velocity
	{
		InputData[3*i]=OriginalData->GetTuple(i+intT*dims[0]*dims[1])[0];
		InputData[3*i+1]=OriginalData->GetTuple(i+intT*dims[0]*dims[1])[1];
		InputData[3*i+2]=OriginalData->GetTuple(i+intT*dims[0]*dims[1])[2];	
		//cout<<"input velocity ("<<InputData[3*i]<<","<<InputData[3*i+1]<<","<<InputData[3*i+2]<<")"<<endl;
	} 
    //get seeds
	int xs=floor((endX-startX)*StepsPerCell);//how many space in x
	int ys=floor((endY-startY)*StepsPerCell);
	cout<<"xs and ys :"<<xs<<", "<<ys<<endl;
	//device_vector<float> d_seeds(xs*ys*3);//seeds of streamline
	float seeds[(1+xs)*(1+ys)*3];
	for(int j=0; j<=ys;j++)
	{
		for(int i=0;i<=xs;i++)
		{
			seeds[(j*(xs+1)+i)*3]=(i/StepsPerCell+startX)*space[0];
			seeds[(j*(xs+1)+i)*3+1]=(j/StepsPerCell+startY)*space[1];
		    seeds[(j*(xs+1)+i)*3+2]=(startT-intT)*space[2];
			cout<<"test initial point "<<seeds[(j*(xs+1)+i)*3]<<" "<<seeds[(j*(xs+1)+i)*3+1]<<" "<<seeds[(j*(xs+1)+i)*3+2]<<endl;
		}	
	}

	float InputSpace[3];
	InputSpace[0]=space[0];
	InputSpace[1]=space[1];
	InputSpace[2]=space[2];

    int InputDims[3];
    InputDims[0]=dims[0];
    InputDims[1]=dims[1];
    InputDims[2]=dims[2];

	float StreamLastPoints[(1+xs)*(ys+1)*3];

    //float* ptrInputSpace=raw_pointer_cast(&d_InputSpace[0]);
    //int* ptrInputSDims=raw_pointer_cast(&d_InputDims[0]);
	//float* ptrInputData=raw_pointer_cast(&d_InputData[0]);
	//float* ptrseeds=raw_pointer_cast(&d_seeds[0]);
	//float* ptrStreamLastPoints=raw_pointer_cast(&d_StreamLastPoints);	

	int cellstep=4;
	float timestep=0.001/(0.04*cellstep);//because every cell 4 steps 0.001/(0.04*4)
	cout<<"xs and ys :"<<xs<<", "<<ys<<endl;
	cout<<"timestep : "<<timestep<<endl;
    int Stoptime=intendT-intstartT-1;
	int step=Stoptime*cellstep;
	cout<<"step : "<<step<<endl;
	int * d_step;
	int * d_xs;
	int * d_ys;
    float * d_timestep;
    int* d_InputDims;
    float* d_InputSpace;
    float* d_InputData;
    float* d_seeds;
    float* d_StreamLastPoints;

    cudaMalloc((void**)&d_timestep, sizeof(float));
    cudaMalloc((void**)&d_step, sizeof(int));
    cudaMalloc((void**)&d_xs, sizeof(int));
    cudaMalloc((void**)&d_ys, sizeof(int));
    cudaMalloc((void**)&d_InputDims, sizeof(int)*3);
    cudaMalloc((void**)&d_InputSpace, sizeof(float)*3);
    cudaMalloc((void**)&d_InputData, sizeof(float)*dims[0]*dims[1]*3*3);
    cudaMalloc((void**)&d_seeds, sizeof(float)*xs*ys*3);
    cudaMalloc((void**)&d_StreamLastPoints, sizeof(float)*xs*ys*3);

    cudaMemcpy(d_step, &step, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, &xs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, &ys, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_timestep, &timestep, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_InputDims, &InputDims,sizeof(int)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_InputSpace, &InputSpace,sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_InputData, &InputData, sizeof(float)*dims[0]*dims[1]*3*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds, &seeds, sizeof(float)*xs*ys*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_StreamLastPoints, &StreamLastPoints, sizeof(float)*xs*ys*3, cudaMemcpyHostToDevice);

	GetStreamLinePoints<<<1,1>>>(d_seeds,d_InputData,d_InputSpace,d_InputDims,step,timestep,
		d_StreamLastPoints,xs,ys);
	//host_vector StreamLineLastPointsResult(d_seeds.size());
	float StreamLineLastPointsResult[xs*ys*3];
	cudaMemcpy(StreamLineLastPointsResult,d_StreamLastPoints,sizeof(float)*xs*ys*3,cudaMemcpyHostToDevice);
	for(int i=0;i<(xs+1)*(ys+1);i++)
	{
		cout<<"the "<<i<<"th point and the last point is ("<<StreamLineLastPointsResult[3*i]<<", "
		<<StreamLineLastPointsResult[3*i+1]<<", "<<StreamLineLastPointsResult[3*i+2]<<")"<<endl;
	}
	cudaFree(d_step);
	cudaFree(d_xs);
	cudaFree(d_ys);
	cudaFree(d_timestep);
	cudaFree(d_StreamLastPoints);
	cudaFree(d_InputDims);
	cudaFree(d_InputSpace);
	cudaFree(d_InputData);
	cudaFree(d_seeds);
    //StreamLineLastPoints=d_StreamLineLastPoints;
    return 0;

}


