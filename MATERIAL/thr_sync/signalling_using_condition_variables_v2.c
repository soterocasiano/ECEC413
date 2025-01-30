/* Example code that shows how to use condition variables to signal between threads. 
 * This code shows how to broadcast a condition variable. 
 *
 * Author: Naga Kandasamy, 04/05/2011
 * Date modified: 10/7/2015
 *
 * Compile as follows: 
 * gcc -o signalling_using_condition_variables_v2 signalling_using_condition_variables_v2.c -std=c99 -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define NUM_WAITING_THREADS 5

typedef struct my_struct_s{
    pthread_mutex_t mutex; /* Protects access to the value. */
    pthread_cond_t condition; /* Signals a change to the value. */ 
    int value; /* The value itself. */
} my_struct_t;

/* Create the data structure and initialize it. */
my_struct_t data = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0}; 

/* Function prototypes. */
void *waiting_thread(void *);
void *signalling_thread(void *);


int 
main(int argc, char **argv)
{
    pthread_t wait_thread_id[NUM_WAITING_THREADS];
    pthread_t signal_thread_id;
    int i;

    /* Create the waiting threads. */
    for(i = 0; i < NUM_WAITING_THREADS; i++)
        pthread_create(&wait_thread_id[i], NULL, waiting_thread, (void *)i);

    /* Create the signalling thread. */
    pthread_create(&signal_thread_id, NULL, signalling_thread, NULL);

    /* Wait to reap the threads that we have created. */
    pthread_join(signal_thread_id, NULL);
    for(i = 0; i < NUM_WAITING_THREADS; i++)
        pthread_join(wait_thread_id[i], NULL);

    pthread_exit(NULL);
}

/* The function executed by the waiting threads. */
void *
waiting_thread(void *args)
{
    int thread_number = (int)args;
    
    pthread_mutex_lock(&data.mutex); 
    /* Check the predicate. */
		  
    if(data.value == 0){
        printf("The value of data is %d. \n", data.value);
        printf("Thread number %d waiting for the condition to be signalled. \n", thread_number);

        /* Release mutex and wait here for condition to be signalled. */
        pthread_cond_wait(&data.condition, &data.mutex); 		  
    }
	  
    /* The cond_wait signal will unblock with the corresponding mutex locked. */	  
    if(data.value != 0){
        printf("Condition signalled to thread number %d. \n", thread_number);
        printf("The value of data = %d. \n", data.value);
    }
		  
    pthread_mutex_unlock(&data.mutex);
    pthread_exit(NULL);
}


/* The function executed by the signalling thread */
void *
signalling_thread(void *args)
{
    int status;
    sleep(5);

    status = pthread_mutex_lock(&data.mutex);
    data.value = 1;
    status = pthread_cond_broadcast(&data.condition); /* Signal the change to all the waiting threads. */
    if(status != 0){
        printf("Error acquiring mutex. \n");
        pthread_exit(NULL);
    }

    status = pthread_mutex_unlock(&data.mutex);
    if(status != 0){
        printf("Error acquiring mutex. \n");
        pthread_exit(NULL);
    }
		 
    pthread_exit(NULL);
}
