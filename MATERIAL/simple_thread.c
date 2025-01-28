/* Program to illustrate basic thread creation and management operations. 
 *
 * Compile as follows: gcc -o simple_thread simple_thread.c -O3 -std=c99 -Wall -pthread
 * Execute as follows: ./simple_thread
 *
 *  Author: Naga Kandasamy
 *  Date created: January 21, 2009
 *  Date modified: January 18, 2024
*/

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

/* Function prototypes for the thread routines. */
void *func_a(void *);
void *func_b(void *);

pthread_t thr_a, thr_b;                         /* Variables that store thread IDs. */
int a = 10;                                     /* Global variable that is stored in the data segment. */

int main(int argc, char **argv)
{
    pthread_t main_thread;
    main_thread = pthread_self();               /* Returns thread ID for the calling thread */
    printf("Main thread = %lu\n", main_thread);

    /* Create new thread and ask it to execute func_a that takes no arguments */
    if ((pthread_create(&thr_a, NULL, func_a, NULL)) != 0) {
        printf("pthread_create error\n");
        exit(EXIT_FAILURE);
    }

    printf("Main thread is thread %lu: Creating thread %lu\n", pthread_self(), thr_a);
    pthread_join(thr_a, NULL);                  /* Wait for thread to finish */

    printf("Value of a is %d\n", a);
    printf("Main thread exiting\n");
    
    pthread_exit((void *)main_thread);
}

/* Function A */
void *func_a(void *arg)
{
    thr_a = pthread_self();                     /* Obtain thread ID */
    int args_for_thread = 5;

    /* Create a new thread and ask it to execute func_b */
    if ((pthread_create(&thr_b, NULL, func_b, (void *)args_for_thread)) != 0) {
        printf("pthread_create error\n");
        exit(EXIT_FAILURE);
    }
		  
    /* Simulate some processing */ 
    printf("Thread %lu is processing\n", thr_a);
    for (int i = 0; i < 5; i++) {
        a = a + 1;
        sleep(1);
    }

    pthread_join(thr_b, NULL);                  /* Wait for thread B to finish */
    
    printf("Thread %lu is exiting\n", thr_a);
    pthread_exit((void *)thr_a);
}

/* Function B */
void *func_b(void *arg)
{
    int args_for_me = (int)arg;
    pthread_t thr_b = pthread_self();
    
    /* Simulate some processing */
    printf("Thread %lu is using args %d\n", thr_b, args_for_me);
    printf("Thread %lu is processing\n", thr_b);
    for (int i = 0; i < 5; i++) {
        sleep(2);
    }

    printf("Thread %lu is exiting\n", thr_b);
    pthread_exit((void *)thr_b);
}







