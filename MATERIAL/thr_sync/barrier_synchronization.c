/* Example code that shows how to use the pthread barrier function.
 * 
 * Compile as follows: gcc -o barrier_synchronization barrier_synchronization.c -std=gnu99 -Wall -lpthread -lm
 * 
 * Author: Naga Kandasamy
 * Date created: February 7, 2022
 * */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>


/* Structure that defines data passed to each thread */
typedef struct thread_data_s {
    int tid;                    /* Thread identifier */
    int num_threads;            /* Number of threads in pool */
    int num_iterations;         /* Number of iterations executed by each thread */
} thread_data_t;

/* Pthread barrier data structure. */
pthread_barrier_t barrier;  

/* Function prototypes */
void *my_thread(void *);

int main(int argc, char **argv)
{   
    if (argc < 3) {
        printf("Usage: %s num-threads num-iterations\n", argv[0]);
        printf("num-threads: Number of threads\n");
        printf("num-iterations: Number of iterations executed by each thread\n");
        exit(EXIT_FAILURE);
    }
		  
    int num_threads = atoi(argv[1]);
    int num_iterations = atoi(argv[2]);

    /* Initialize the barrier data structure */
    pthread_barrier_init(&barrier, NULL, num_threads);

    /* Create the threads */
    int i;
    pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);

    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_iterations = num_iterations;
        pthread_create(&tid[i], NULL, my_thread, (void *)&thread_data[i]);
    }

    /* Wait for threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);
	  
    pthread_exit(NULL);
}

/* Function executed by each thread */
void *my_thread(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;

    int i;    
    for (i = 0; i < thread_data->num_iterations; i++) {
        printf("Thread number %d is processing for iteration %d\n", thread_data->tid, i);
        
        sleep(ceil(rand()/(float)RAND_MAX * 10)); /* Simulate some processing */

        printf("Thread %d is at synchronization barrier\n", thread_data->tid);

        /* Wait here for all threads to catch up before starting the next iteration. */
        pthread_barrier_wait(&barrier);     
    }
    
    pthread_exit(NULL);
}


