#include<stdio.h>
#include<math.h>

// defining the function LogSumExp  
double logsumexp(double nums[]) {
  double max_exp = nums[0], sum = 0.0;
    for (int i = 1 ; i < 4 ; i++){
        // printf(" num[%d]:  %f ",i, nums[i]);
        if (nums[i] > max_exp){
            max_exp = nums[i];
        }
    }
    // printf(" max_exp:  %f ", max_exp);
  for (int i = 0; i < 4 ; i++){
      sum += exp(nums[i] - max_exp);
  }
  return log(sum) + max_exp;
}

// function for printing new line - just for easy coding
void newline(){
    printf("\n");
}

// Start of Main Function
int main(){
    // Six Examples taken for train binary features (five features) and classified into 4 classes [0,1,2,3]
    int trainFeature[6][5] = { 
        {1,0,0,1,0},
        {0,1,0,1,0},
        {1,0,0,1,0},
        {0,0,1,0,1},
        {1,1,0,1,1},
        {0,1,1,1,1}
    };
    int trainCLass[6]={0,1,3,1,2,0};
    int maxm_c;
    double probability[4][5][2] = {0};
    double classProbability[4]={0};
    double LSP; //LSP - LogSumExp
    double maxProbability;
    double maxm;
    double normalizer;
    
    //Test data
    int testFeature[4][5] = {
        {1,1,1,1,1},
        {0,0,0,0,0},
        {1,0,0,0,1},
        {0,0,0,1,1}
    };
    float testProbability[4][4]={0};
    double Log_testProbability[4][4]={0};
    
    //Displaying Training data
    printf("Train data \n");
    for(int i=0;i<6;i++){
        for(int j=0;j<5;j++){
            printf("%d  ",trainFeature[i][j]);
        }
        newline();  
    }
    
    //Displaying train classes
    printf("trainCLass \n");
    for(int i=0;i<6;i++){
        printf("%d  ",trainCLass[i]);
    }
    newline();
    
    
    
    printf("----------------------------------------------------------------------------------------------------");
    
    //Calculating the counting of occurence of event
    for(int j=0;j<5;j++){
        for(int k=0;k<6;k++){
                probability[trainCLass[k]][j][trainFeature[k][j]]++;
        }
    }
    
    
    //Calculating the probability of each features with values 0 or 1 with given class i.e Table of p(x[i]=0 or 1 | c = {0,1,2,3}) where i=1:5
    for(int i=0;i<4;i++){
        for(int j=0;j<5;j++){
            normalizer = probability[i][j][0]+probability[i][j][1];
            for(int k=0;k<2;k++){
                probability[i][j][k] = probability[i][j][k]/normalizer;
            }
        }
        newline();
    }
    
    //Displaying the Table
    printf("Probability Table \n");
    for(int i=0;i<4;i++){
        for(int j=0;j<5;j++){
            for(int k=0;k<2;k++){
                printf("%f  ",probability[i][j][k]);
            }
            printf("      ");
        }
        printf("\n");
    }
    newline();
    
        //Calculating class probability
    for(int j=0;j<6;j++){
        classProbability[trainCLass[j]]++;
    }
    
    for(int j=0;j<4;j++){
        classProbability[j] = classProbability[j]/6; // 6 is the total number of training examples
    }
    
    //Displaying the calculated class probability
    printf("classProbability - We have 4 classes {0,1,2,3} \n");
    for(int i=0;i<4;i++){
        printf(" %f ", classProbability[i]);
    }
    newline();
    printf("----------------------------------------------------------------------------------------------------");
    newline(0);
    newline(0);
    
    printf("------------------------------Testing the Naive Bayes Algorithms------------------------------------");
    
    newline();
    newline();
    //Displaying the test data
    newline();
    printf("Test Data \n");
    for(int i=0;i<4;i++){
        for(int j=0;j<5;j++){
            printf("%d  ",testFeature[i][j]);
        }
        newline();  
    }
    
    // Now Testing the NBC - Naive Bayes Classifier - Refer to the Naive Bayes algorithm in paper
    
    // -------------------------- START OF THE ALGORITHM ------------------------------------------
    for(int i=0;i<4;i++){
        for(int c=0;c<4;c++){
            // printf(" Log_testProbability[%d][%d]: %f ",i,c,Log_testProbability[i][c]);
            Log_testProbability[i][c] = log(classProbability[c]);
            // printf(" Log_testProbability[%d][%d]: %f ",i,c,Log_testProbability[i][c]);
            // newline();
            // newline();
            for(int j=0;j<5;j++){
                if(testFeature[i][j]==1){
                    // printf("Inside %d features with values 1",j);
                    // printf(" Log_testProbability[%d][%d]: %f ",i,c,Log_testProbability[i][c]);
                    // newline();
                    // printf(" probability[%d][%d][%d] : %f ",i,c,testFeature[i][j],probability[c][j][1]);
                    // printf(" log(probability[%d][%d][%d]) : %f",i,c,testFeature[i][j], log(probability[c][j][1]));
                    Log_testProbability[i][c] = Log_testProbability[i][c] + log(probability[c][j][1]);
                    // printf("\nAfter calculation\n");
                    // printf(" Log_testProbability[%d][%d]: %f ",i,c,Log_testProbability[i][c]);
                    // newline();
                }
                else{
                    Log_testProbability[i][c] = Log_testProbability[i][c] + log(1-probability[c][j][0]); 
                    // newline();
                    // printf("Inside %d features with values 0",j);
                    // printf(" Log_testProbability[%d][%d]: %f ",i,c,Log_testProbability[i][c]);
                }
            }
            // newline();
            // newline();
        }
        // newline();
        printf("----------------------------------------------------------------------------------------------------\n");
        newline();
        printf("Printing the calculated probablility of each class for test data - %d \n", i);
        for(int k=0;k<4;k++){
            printf(" Log_testProbability[%d][%d] %f \n",i,k,Log_testProbability[i][k]);
        }
        
        //Calculating the LogSumExp using logsumexp function
        LSP = logsumexp(Log_testProbability[i]);
        printf(" \n logsumexp[%d] : %f \n",i,LSP);
        
        //Displaying testProbability
        newline();
        // printf("----------------------------------------------------------------------------------------------------\n");
        printf("\n testProbability for data [%d] \n", i);
        
        // Displaying the predicted probablity of each class for each test data - we have taken 4 test data
        newline();
        // printf("----------------------------------------------------------------------------------------------------\n");
        for(int c=0;c<4;c++){
            testProbability[i][c]=exp(Log_testProbability[i][c]-LSP);
            printf(" %f ", testProbability[i][c]);
            // maxm=0.0;
            if(maxm<testProbability[i][c]){
                maxm=testProbability[i][c];
                maxm_c = c;   // this is the argmax-maxm
            }
        }
        printf(" \t predicted class : %d\n", maxm_c);
        // newline();
        // newline();
    }
    
    // -------------------------- END OF THE ALGORITHM ------------------------------------------
    
    
}



