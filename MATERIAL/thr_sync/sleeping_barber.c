/* This code illustrates the sleeping barber problem. 
 * 
 * Compile as follows:
 * gcc -o sleeping_barber sleeping_barber.c -std=c99 -Wall -pthread -lm
 *
 * Author: Naga Kandasamy
 * Date modified: April 23, 2020

*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <math.h>

#define TRUE 1
#define FALSE 0
#define MIN_TIME 4
#define MAX_TIME 8

#define MAX_NUM_CUSTOMERS 50

/* Function prototypes */
void *customer(void *num); 
void *barber(void *); 
int UD(int, int); 

/* Definition of semaphores */
sem_t waiting_room;                         /* Signal that waiting room can accommodate customers */
sem_t barber_seat;                          /* Signal to ensure exclusive access to barber seat */
sem_t done_with_customer;                   /* Signal customer that barber is done with him/her */
sem_t barber_bed;                           /* Signal to wake up barber */

int done_with_all_customers = FALSE;        /* Flag indicating barber can go home */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s num-customers num-chairs\n", argv[0]);
        fprintf(stderr, "num-customers: maximum number for customers than barber will process\n");
        fprintf(stderr, "num-chairs: number of chairs in the waiting room\n");
        exit(EXIT_FAILURE);
    }

    int num_customers = atoi(argv[1]);        /* Number of customers */
    int num_waiting_chairs = atoi(argv[2]);   /* Number of waiting chairs in the barber shop */
  
    srand(time(NULL));
  
    if (num_customers > MAX_NUM_CUSTOMERS) {
        fprintf(stderr, "Number of customers exceeds maximum capacity of the barber\n");
        fprintf(stderr, "Resetting the number of customers to %d\n", MAX_NUM_CUSTOMERS);
        num_customers = MAX_NUM_CUSTOMERS;
    }
  
    sem_init(&waiting_room, 0, num_waiting_chairs);  /* Initialize semaphores */
    sem_init(&barber_seat, 0, 1);
    sem_init(&done_with_customer, 0, 0);
    sem_init(&barber_bed, 0, 0);
  
    pthread_t btid; /* ID for the barber thread */
    pthread_t tid[MAX_NUM_CUSTOMERS]; /* IDs for customer threads */

    pthread_create(&btid, 0, barber, 0); /* Create barber thread */
    /* Create customer threads and give each an ID */
    int customerID[MAX_NUM_CUSTOMERS]; 
    int i;
    for (i = 0; i < num_customers; i++) {
        customerID[i] = i;
        pthread_create(&tid[i], 0, customer, &customerID[i]);
    }
  
    for (i = 0; i < num_customers; i++)
        pthread_join(tid[i], 0);
  
    done_with_all_customers = TRUE;
    sem_post(&barber_bed); /* Wake up barber */
    
    pthread_join(btid, 0);
    exit(EXIT_SUCCESS);
}

/* Barber thread */
void *barber(void *arg)
{
    int wait_time;
  while (!done_with_all_customers) {
    fprintf(stderr, "Barber: Sleeping\n");
    sem_wait(&barber_bed);

    if (!done_with_all_customers) {
      fprintf(stderr, "Barber: Cutting hair\n");
      wait_time = UD(MIN_TIME, MAX_TIME); /* Simulate cutting hair */
      sleep(wait_time);
      sem_post(&done_with_customer); /* Indicate that chair is free */
    }
    else {
      fprintf(stderr, "Barber: Done for the day\n");
    }
  }
  pthread_exit(NULL);
}

/* Customer thread */
void *customer(void *customer_number)
{
  int number = *(int *)customer_number;
  fprintf(stderr, "Customer %d: Leaving for the barber shop\n", number);
  int wait_time = UD(MIN_TIME, MAX_TIME); /* Simulate going to barber shop */
  sleep(wait_time);
  fprintf(stderr, "Customer %d: Arrived at the barber shop\n", number);

  sem_wait(&waiting_room); /* Wait to get into the barber shop */
  fprintf(stderr, "Customer %d: Entering waiting room\n", number);

  sem_wait(&barber_seat); /* Wait for barber to become free */
  sem_post(&waiting_room); /* Let people waiting outside the shop know */

  sem_post(&barber_bed); /* Wake up barber */
  sem_wait(&done_with_customer); /* Wait until hair is cut */
  sem_post(&barber_seat); /* Give up seat */

  fprintf(stderr, "Customer %d: Going home\n", number);
  pthread_exit(NULL);
}

/* Returns a random number between min and max */
int UD(int min, int max)
{
  return ((int)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX))));
}




