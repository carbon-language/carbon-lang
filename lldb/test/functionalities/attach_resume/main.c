#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <pthread.h>

void *start(void *data)
{
    int i;
    size_t idx = (size_t)data;
    for (i=0; i<30; i++)
    {
        if ( idx == 0 )
            usleep(1);
        sleep(1);
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    static const size_t nthreads = 16;
    pthread_attr_t attr;
    pthread_t threads[nthreads];
    size_t i;

    pthread_attr_init(&attr);
    for (i=0; i<nthreads; i++)
        pthread_create(&threads[i], &attr, &start, (void *)i);

    for (i=0; i<nthreads; i++)
        pthread_join(threads[i], 0);
}
