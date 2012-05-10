#include <pthread.h>
#include <unistd.h>

volatile int g_data1;
volatile int g_data2;
volatile int g_data3;
volatile int g_data4;

void *Thread1(void *x) {
  if (x)
    usleep(1000000);
  g_data1 = 42;
  g_data2 = 43;
  g_data3 = 44;
  g_data4 = 45;
  return 0;
}

int main() {
  pthread_t t;
  pthread_create(&t, 0, Thread1, (void*)1);
  Thread1(0);
  pthread_join(t, 0);
}

// CHECK: ThreadSanitizer: reported 1 warnings
