/* Example code that shows how to use condition variables to signal between threads. 
 * Author: Naga Kandasamy, 04/05/2011
 * Date modified: 10/7/2015
 *
 * Compile as follows: 
 * gcc -o signalling_using_condition_variables_v1 signalling_using_condition_variables_v1.c -std=c99 -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

typedef struct my_struct_s{
    pthread_mutex_t mutex; /* Protects access to the value. */	  
    pthread_cond_t condition; /* Signals a change to the value. */ 
    int value; /* The value itself. */
} my_struct_t;

/* Create the data structure and initialize it. */
my_struct_t data = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0}; 

/* Function prototypes. */
void *my_thread(void *);


int 
main(int argc, char **argv)
{
    pthread_t thread_id;
    int status; 

    status = pthread_create(&thread_id, NULL, my_thread, NULL);
    if(status != 0){
        printf("Error creating thread. \n");
        pthread_exit(NULL);
    }

    /* Wait on the condition variable to be changed by the thread */
    status = pthread_mutex_lock(&data.mutex);
    if(status != 0){
        printf("Error acquiring mutex. \n");
        pthread_exit(NULL);
    }

    /* Test the predicate, that is the data and wait for a condition to become true if neccessary. */ 
    if(data.value == 0){
        printf("Value of data = %d. Waiting for someone to change it and signal me. \n", data.value);

        /* Release the mutex and wait for a thread to signal the condition. */
        status = pthread_cond_wait(&data.condition, &data.mutex); 					 
        if(status != 0){
            printf("Error waiting on condition. \n");
            pthread_exit(NULL);
        }
    }

    /* The cond_wait call always returns with the mutex locked. */
    if(data.value != 0){
        printf("Condition was signalled. \n");
        printf("Value of data = %d. \n", data.value);
    }
    status = pthread_mutex_unlock(&data.mutex);

    pthread_join(thread_id, NULL);
    pthread_exit(NULL);
}

/* The function executed by the thread. */
void *
my_thread(void *args)
{
    int status;
    sleep(5);
    status = pthread_mutex_lock(&data.mutex);
    if(status != 0){
        printf("Error acquiring mutex. \n");
        pthread_exit(NULL);
    }

    data.value = 1;
    status = pthread_cond_signal(&data.condition); // Signal the change in state to the waiting threads 	  
    if(status != 0){
        printf("Error signalling condition. \n");
        pthread_exit(NULL);
    }
		  
    status = pthread_mutex_unlock(&data.mutex);
    if(status != 0){
        printf("Error releasing mutex. \n");
        pthread_exit(NULL);
    }

    pthread_exit(NULL);
}
