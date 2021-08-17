
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<stdbool.h>

const int q=40;    // Number of rows of input data
const int c=2;     // Number of class for classification

float X[][2]= {{-9.8205,	6.6610},
    {2.2405,	3.6145},
    {0.5950,	6.4402},
    {2.5644,	-10.1175},
    {6.1184,	-1.9583},
    {-7.8832,	0.0027},
    {2.0000,	-4.0000},
    {2.4875,	4.65181},
    {-4.6008,	12.4886},
    {-3.2508,	9.8378},
    {2.6028,	4.3551},
    {0.9928,	-4.2604},
    {1.8607,	-5.9187},
    {-5.5412,	-0.5394},
    {0.5894,	2.9496},
    {-4.2404,	10.1937},
    {5.3800,	-1.7036},
    {-3.0000,	5.0000},
    {-8.7283,	2.2468},
    {-8.2287,	0.8988},
    {-8.4336,	1.2946},
    {2.7590,	1.0784},
    {2.9531,	-6.9680},
    {6.1452,	-0.7751},
    {2.7412,	5.1291},
    {-2.0094,	-3.5032},
    {-1.5771,	-1.6737},
    {3.1597,	0.4768},
    {-6.5813,	-0.9902,},
    {-3.2587,	8.1883},
    {-4.4487,	1.3225},
    {1.6823,	6.2172},
    {4.8578,	-0.2035},
    {-8.8289,	1.3517},
    {-7.8230,	4.3950},
    {-2.3123,	-4.2688},
    {-3.3079,	8.7399},
    {-1.9877,	-1.1538},
    {-7.6989,	0.3054},
    {-0.9809,	4.1680}
};
float y[]= {-1,1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,1};

float dot_product(float A[], float B[]);
float SumDiff(float A[], float B[]);
float SumSquareDiff(float A[], float B[]);
void Linear_kernel(float X[][c], float K[][q]);
void Polynomial_kernel(float X[][c], float K[][q], int m, int p);
void Laplacian_kernel(float X[][c], float K[][q], int lambda);
void Gaussian_kernel(float X[][c], float K[][q], float lambda, float sigma);
void Sigmoid_kernel(float X[][c], float K[][q], float m, float gamma);
//float* SMO(float X[][c], float K[][q], float C, float tol,int max_pass);
float Error(int i, float b,float K[][q], float alpha[]);
float max(float a, float b);
float min(float a, float b);
float absolute(float a);
//------------------------------------------------------------ Main Function -----------------------------------------------------------------------------
int main()
{
    int i;
    int j;
    float old_alpha_i, old_alpha_j;
    float K[q][q];   // Kernel Matrix
    float HM[q][q];  // Hessian Matrix
    float alpha[q];
    float b=0, b1, b2;
    float E[q];    // Error array
    float L=0, H=0;
    float eta=0;
    int num_changed_alpha;
    int passes=0;
    int max_pass=20;
    float tol=1e-3;
    float C=10;


    printf("C: %f\tL:%f\tH:%f",C,L,H);
    //Initialize Lagrange Multipliers with zero
    for(i=0; i<q; i++)
    {
        alpha[i]=0.0;
    }

    // Computing Kernel Matrix (using Linear Kernel)
    Linear_kernel(X, K);

    // Displaying Kernel Matrix
    for(i=0; i<q; i++)
    {
        for(j=0; j<q; j++)
        {
            printf(" %f \t", K[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Computing Kernel Matrix
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            HM[i][j]=y[i]*y[j]*K[i][j];
            HM[j][i]=HM[i][j];
        }
    }

    // ALPHA=SMO(X, K, 10, .01, 20);

    // ----------------SMO Algorithm code (inside main function) ---------------------------------------
    while(passes<max_pass)
    {
        num_changed_alpha=0;
        for(i=0; i<q; i++)
        {
            E[i]=Error(i,b,K,alpha);
            printf("\nE[%d]: %f\n",i,E[i]);
            if((y[i]*E[i]<-tol && alpha[i]<C) || (y[i]*E[i]>tol && alpha[i]>0))
            {
                j=(rand()%q);
                printf("\nj: %d\n",j);
                E[j]=Error(j,b,K,alpha);
                old_alpha_i=alpha[i];
                old_alpha_j=alpha[j];

                //Compute L and H
                printf("C: %f\tL:%f\tH:%f\n",C,L,H);
                if(y[i]==y[j])
                {
                    L = max(0, alpha[j]+alpha[i]-C);
                    H = min(C, alpha[j]+alpha[i]);
                    printf("L=max(0, %f)\tH=min(%f,%f)\n", alpha[j]+alpha[i]-C, C, alpha[j]+alpha[i]);
                    printf("L: %f\tH: %f\n", L,H);
                }
                else
                {
                    L = max(0, alpha[j]-alpha[i]);
                    H = min(C, C+alpha[j]-alpha[i]);
                }
                printf("\nL: %f\tH: %f\n", L,H);
                if(L==H)
                {
                    continue;
                }

                // Compute eta(n)
                eta=2*K[i][j]-K[i][i]-K[j][j];
                printf("\neta: %f\n",eta);
                if(eta>=0)
                {
                    continue;
                }

                alpha[j] = alpha[j] - (y[j]*(E[i]-E[j]))/eta;
                if(alpha[j]>H)
                {
                    alpha[j]=H;
                }
                if(alpha[j]<L)
                {
                    alpha[j]=L;
                }
                //printf("i: %d",i);
                //printf("alpha[%d]: %f\told_alpha[%d]: %f\nalpha[%d]: %f\told_alpha[%d]: %f", i,alpha[i],old_alpha_i,j,alpha[j],old_alpha_j);
                //printf(abs(alpha[i]-alpha[j])<tol ? "TRUE" : "FALSE");
                if(absolute(alpha[i]-alpha[j])<tol)
                {
                    alpha[j]=old_alpha_j;
                    continue;
                }

                // Determine alpha[i]
                alpha[i] = alpha[i] + y[i]*y[j]*(old_alpha_j-alpha[j]);

                // Compute b1 and b2
                b1 = b - E[i] - y[i]*(alpha[i]-old_alpha_i)*K[i][i] - y[j]*(alpha[j]-old_alpha_j)*K[i][j];
                b2 = b - E[j] - y[i]*(alpha[i]-old_alpha_i)*K[i][j] - y[j]*(alpha[j]-old_alpha_j)*K[j][j];


                // Compute b
                if(alpha[i]>0 && alpha[i]<C)
                {
                    b=b1;
                }

                else if(alpha[j]>0 && alpha[j]<C)
                {
                    b=b2;
                }

                else
                {
                    b=(b1+b2)/2;
                }

                num_changed_alpha = num_changed_alpha + 1;

            }
        }

        if(num_changed_alpha==0)
        {
            passes = passes + 1;
        }
        else
        {
            passes = 0;
        }
    }


    for(i=0; i<q; i++)
    {
        printf(" %f \t", alpha[i]);
    }

    return 0;
}
//---------------------------------------------------------- End of Main Function --------------------------------------------------------------------------------

float dot_product(float A[], float B[])    // passing argument code - dot_product(X[0], X[1])
{
    float sum=0;
    int i;
    for(i=0; i<c; i++)
    {
        sum = sum + A[i]*B[i];
    }
    return sum;
}

// SumDiff
float SumDiff(float A[], float B[])
{
    float sum=0.0;
    float diff;
    int i;
    for(i=0; i<c; i++)
    {
        diff = A[i]-B[i];
        sum= sum+diff;
    }
    printf("\n sum: %f\n", sum);
    return sum;
}

float SumSquareDiff(float A[], float B[])
{
    float sum=0;
    float diff;
    int i;
    for(i=0; i<c; i++)
    {
        diff = A[i]-B[i];
        sum= sum+pow(diff,2);
    }
    return sum;
}

// --------------------------------------------------- KERNEL FUNCTION ---------------------------------------------------------------------------------------------

void Linear_kernel(float X[][c], float K[][q])
{
    int i,j;
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            K[i][j]=dot_product(X[i],X[j]);
            K[j][i]=K[i][j];
        }
    }

}

void Polynomial_kernel(float X[][c], float K[][q], int m, int p)
{
    float dot;
    float d;
    int i,j;
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            dot = dot_product(X[i],X[j]);
            d = dot+m;
            K[i][j]=pow(d,p);
            K[j][i]=K[i][j];
        }
    }
}

void Laplacian_kernel(float X[][c], float K[][q], int lambda)
{
    float sum_diff;
    int i,j;
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            sum_diff = SumDiff(X[i], X[j]);
            K[i][j]=exp(-lambda*sum_diff);
            K[j][i]=K[i][j];
        }
    }
}

void Gaussian_kernel(float X[][c], float K[][q], float lambda, float sigma)
{
    float sum_sqr_diff;
    int i,j;
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            sum_sqr_diff = SumSquareDiff(X[i], X[j]);  // Write code for this function
            K[i][j]=exp(-lambda*sum_sqr_diff/(2*pow(sigma,2)));
            K[j][i]=K[i][j];
        }
    }
}

void Sigmoid_kernel(float X[][c], float K[][q], float m, float gamma)
{
    float dot;
    int i,j;
    for(i=0; i<q; i++)
    {
        for(j=0; j<=i; j++)
        {
            dot = dot_product(X[i], X[j]);
            K[i][j]=tanh(gamma*dot+m);  //Write code for tanh function
            K[j][i]=K[i][j];
        }
    }
}


//--------------------------------------------------- END OF KERNEL FUNCTION --------------------------------------------------------------------------------------------------

float Error(int i, float b,float K[][q], float alpha[])
{
    float sum=0.0;
    float temp;
    int j;

    for(j=0; j<q; j++)
    {
        temp=alpha[i]*y[i]*K[i][j];
        sum=sum+temp;
    }
    return b+sum-y[i];
}

float max(float a, float b)
{
    return (a>b)? a: b;
}

float min(float a, float b)
{
    return (a>b)? b: a;
}

float absolute(float a)
{
    return (a>0)? a:-a;
}
