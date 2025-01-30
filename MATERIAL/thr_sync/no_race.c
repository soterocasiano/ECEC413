/* Program to illustrate how to avoid race conditions by protecting 
 * shared variables using locks. 
 *
 * Author: Naga Kandasamy
 * Date created: February 10, 2020
 *
 * Compile as follows: gcc -o no_race no_race.c -std=c99 -O3 -Wall -pthread -lm
 *
 * Usage: ./no_race num-threads 
*/

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Function prototype for the thread routines */
void *worker(void *);

/* Global variable that is shared among threads */
long int sum = 0;
/* Mutex to protect shared variable */
pthread_mutex_t mutex;

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s num-threads\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    unsigned int num_threads;
    num_threads = atoi(argv[1]); 
    
	srand(time(NULL));

    /* Initialize mutex */
    pthread_mutex_init(&mutex, NULL);

    /* Fork point: create worker threads */
    pthread_t *worker_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));	
    int i;
    for (i = 0; i < num_threads; i++) {
        if ((pthread_create(&worker_thread[i], NULL, worker, NULL)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
		  
    /* Join point: wait for all the worker threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(worker_thread[i], NULL);

    printf("Final sum = %ld\n", sum);
		
    /* Free data structures */
    free((void *)worker_thread);
    pthread_exit(NULL);
}


/* Function that will be executed by all the worker threads */
void *worker(void *arg)
{
    int i, j, num_iter;

    /* Simulate some processing */
    for (i = 0; i < 1000000; i++) {
        num_iter = ceil(100000000 * (float)rand()/RAND_MAX);
        for (j = 0; j < num_iter; j++);
  
        /* Increment global sum. Acquire the mutex first. */
        pthread_mutex_lock(&mutex);
        sum += 1;
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit(NULL);
}







