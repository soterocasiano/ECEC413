/* This code illustrates how to create a signal handler thread.
 *
 * Compile as follows: gcc -o signal_handler_thread signal_handler_thread.c -O3 -std=c99 -Wall -pthread
 * Execute as follows: ./signal_handler_thread
 *
 * Author: Naga Kandasamy, 
 * Date created: January 22, 2009
 * Date modified: April 16, 2020 
*/

#define _POSIX_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0

void *signal_handler(void *); /* The signal handler prototype */

int main(int agrc, char **argv)
{
    sigset_t set; /* Define the set of signals */
    pthread_t thread_id;
  
    /* Block all signals in the main thread. Threads created after this will 
     * inherit this, i.e, also block all signals. */
    sigfillset(&set);
    sigprocmask(SIG_BLOCK, &set, NULL);

    /* Create a signal_handler thread to handle all non-directed signals */
    if ((pthread_create(&thread_id, NULL, signal_handler, NULL)) != 0) {			 
        fprintf(stderr, "Cannot create signal handler thread\n");			 
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Main thread processing\n");
    while (TRUE) {
        /* Simulate some processing in the main thread. */
    }

    exit(EXIT_SUCCESS);
}

/* The signal handler thread */
void *signal_handler(void *arg)
{
    sigset_t set;
    int sig;

    sigfillset(&set); /* Catch all signals */
    while (TRUE) {
        /* Wait for some signal to arrive and handle the signal appropriately */
        if (sigwait(&set, &sig) != 0) {
            fprintf(stderr, "Error waiting on a signal\n");
            break;
        };
			
        switch (sig) {
            case SIGINT: 
                fprintf(stderr, "Ctrl+C received...quiting\n");		
                exit(EXIT_SUCCESS);

            case SIGUSR1: 
                fprintf(stderr, "Received SIGUSR1 signal\n");
                break;

            case SIGUSR2:
                fprintf(stderr, "Received SIGUSR2 signal\n");
                break;

            default:
                fprintf(stderr, "Received some signal\n");	 
        } /* End switch */
    } /*End while */
  
    return((void *)0);
}
