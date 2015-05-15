#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void *
thread3 (void *input)
{
    pthread_mutex_lock(&mutex);
    pthread_cond_signal(&cond); // Set break point at this line. 
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void *
thread2 (void *input)
{
    pthread_mutex_lock(&mutex);
    pthread_cond_signal(&cond);
    pthread_cond_wait(&cond, &mutex);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void *
thread1 (void *input)
{
    pthread_t thread_2;
    pthread_create (&thread_2, NULL, thread2, NULL);

    pthread_join(thread_2, NULL);

    return NULL;
}

int main ()
{
    pthread_t thread_1;
    pthread_t thread_3;

    pthread_mutex_lock (&mutex);

    pthread_create (&thread_1, NULL, thread1, NULL);

    pthread_cond_wait (&cond, &mutex);

    pthread_create (&thread_3, NULL, thread3, NULL);

    pthread_cond_wait (&cond, &mutex);

    pthread_mutex_unlock (&mutex);

    pthread_join (thread_1, NULL);
    pthread_join (thread_3, NULL);

    return 0;
}
