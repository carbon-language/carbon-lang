#ifndef MY_SLEEP_H
#define MY_SLEEP_H

#include <stdio.h>
#include<stdlib.h>
#include<unistd.h>

#include <sys/times.h> 
#include <sys/time.h>
#include <time.h>
#include <errno.h>

/*! Utility function to have a sleep function with better resolution and which only stops one thread. */

static void my_sleep(double sleeptime){
  struct timeval tv;
  struct timezone tzp;
  double start;
  double real;
  if(gettimeofday(&tv,&tzp)!=0) {
    perror("get_time: ");
    exit(-1);
  }
  start = (double)tv.tv_sec + ((double)tv.tv_usec/1000000.0);
  real=start;
  while( (real-start)<sleeptime){
    if(gettimeofday(&tv,&tzp)!=0) {
      perror("get_time: ");
      exit(-1);
    }
    real = (double)tv.tv_sec + ((double)tv.tv_usec/1000000.0);
  }
}

#endif
