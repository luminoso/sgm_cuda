
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

// includes, project
#include <cutil_inline.h>

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-v] image1 image2\n",command);
}

// main
int main( int argc, char** argv) 
{
    bool verbose=false;  // used to print the coordinates of each pixel that is different

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"hv")) !=-1)
    {
        switch(opt)
        {
            case 'h':
                usage(argv[0]);
                exit(0);
                break;
            case 'v':
                verbose=true;
                break;
        }
    }

    if(optind>=argc-1) { // should still have 2 arguments after options
                usage(argv[0]);
                exit(0);
    }

    // allocate host memory 1
    unsigned int* h_idata1=NULL;
    unsigned int h1,w1;
    //load pgm
    if (cutLoadPGMi(argv[optind], &h_idata1, &w1, &h1) != CUTTrue) {
        printf("Failed to load image file: %s\n", argv[1]);
        exit(1);
    }

    // allocate host memory 2
    unsigned int* h_idata2=NULL;
    unsigned int h2,w2;
    //load pgm
    if (cutLoadPGMi(argv[optind+1], &h_idata2, &w2, &h2) != CUTTrue) {
        printf("Failed to load image file: %s\n", argv[2]);
        exit(1);
    }

    if(h1!=h2 || w1!=w2) {
         printf("images differ in size %ux%u vs. %ux%u\n", w1, h1, w2,h2);
         exit(0);
    }

    int histDiffs[6]={0,0,0,0,0,0};  // to save histogram of differences; 
                                     // histDiffs[0] stores diff of 1; histDiffs[1] stores diff of 2; ...
                                     // histDiffs[5] stores diffs larger than 5
    bool diffs=false;

    // compute histogram
    for(int i=0; i<w1*h1; i++) {
         int diff = abs((int)h_idata1[i]-(int)h_idata2[i]);
         if(diff>0) {
              if(verbose) {
                  printf("diff at %d,%d %d!=%d\n", i%w1, i/w1, h_idata1[i], h_idata2[i]);
              }
              if(diff<6) {
                  histDiffs[diff-1]++;
              }
              else {
                  histDiffs[5]++;
              }
              diffs = true;
         }
    }

    if(!diffs) {
            printf("images are identical\n");
    }
    else {
          // print histogram
          for(int d=1; d<6; d++) {
             printf("%d diffs of %d\n",histDiffs[d-1],d);
          }
          printf("%d diffs > 5\n",histDiffs[5]);
    }

    return 0;
}
