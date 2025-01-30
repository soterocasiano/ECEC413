/* Compile as follows: 
   gcc -o dining_philosophers -std=c99 dining_philosophers.c -std=c99 -Wall -pthread -lm -lrt
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>

#define NUM_PHILOSOPHERS 5
#define MAX_NUM_MEALS 20
#define MIN_TIME 2
#define MAX_TIME 5

void *philosopher(void *); /* Philosopher function */
int UD(int, int); /* Function returns a random number between min and max */

/* Status of each chopstick. The status flag is 1 if the chopstick is free, 0 if in use */
int chopstick[NUM_PHILOSOPHERS] = {1, 1, 1, 1, 1};

/* Mutex locks for each chopstick */
pthread_mutex_t chopstick_lock[NUM_PHILOSOPHERS];

/* Philodsopher IDs */
int philosopher_ID[NUM_PHILOSOPHERS] = {0, 1, 2, 3, 4};

/* Meals consumed by each philosopher */
int num_meals_consumed[NUM_PHILOSOPHERS] = {0, 0, 0, 0, 0};
int curr_num_meals = 0; /* Number of meals eaten */
pthread_mutex_t curr_num_meals_lock;

/* Main program */
int main(){
  pthread_t tid[NUM_PHILOSOPHERS]; /* Thread IDs corresponding to each phil. */

  srand((long)time(NULL)); /* Initialize randomizer */

  int i;
  void *status;
  /* Create NUM_PHILOSOPHERS philosophers */
  for(i = 0; i < NUM_PHILOSOPHERS; i++){
    if((pthread_create(&tid[i], 0, philosopher, &philosopher_ID[i])) != 0){
      printf("Error creating threads \n");
      exit(1);
    }
  }

  /* Wait for NUM_PHILOSOPHER threads to join the main thread */
  for(i = 0; i < NUM_PHILOSOPHERS; i++){
    if((pthread_join(tid[i], &status)) != 0){
      printf("Error during thread join \n");
      exit(1);
    }
  }

  /* Print statistics */
  printf("\n");
  for(i = 0; i < NUM_PHILOSOPHERS; i++)
    printf("Philosopher %d ate %d meals \n", 
	   philosopher_ID[i], num_meals_consumed[i]);

  printf("Main thread exiting \n");
  return(0);
}

/* Philosopher function */
void *philosopher(void *arg){
  int curr = *((int *)arg);
  while(curr_num_meals < MAX_NUM_MEALS){
    printf("Philosopher %d: Trying to eat \n", curr);
    
    pthread_mutex_lock(&chopstick_lock[curr]); /* Try to lock left chopstick */
    if(chopstick[curr] == 1){ /* Left chop stick is available */
      printf("Philosopher %d: Got left chopstick \n", curr);
      chopstick[curr] = 0; /* Set chopstick status to in use */
      pthread_mutex_unlock(&chopstick_lock[curr]); /* Unlock left chopstick */
      
      pthread_mutex_lock(&chopstick_lock[(curr + 1) % NUM_PHILOSOPHERS]); /* Lock right stick */

      if(chopstick[(curr + 1) % NUM_PHILOSOPHERS] == 1){ /* Available */
	printf("Philosopher %d: Got right chopstick \n", curr);
	chopstick[(curr + 1) % NUM_PHILOSOPHERS] = 0; /* Set status to in use */
	pthread_mutex_unlock(&chopstick_lock[(curr + 1) % NUM_PHILOSOPHERS]);
	
	/* Eat for some random time */
	printf("Philosopher %d: GOT BOTH CHOPSTICKS! \n", curr);
	num_meals_consumed[curr]++; /* Increment number of meals consumed by curr */
	pthread_mutex_lock(&curr_num_meals_lock);
	curr_num_meals++; /* Increment total number of meals consumed */
	pthread_mutex_unlock(&curr_num_meals_lock);
	int eatTime = UD(MIN_TIME, MAX_TIME);
	printf("Philosopher %d: Will eat for %d seconds \n", curr, eatTime);
	sleep(eatTime);

	/* Give up the chopsticks */
	printf("Philosopher %d: Giving up left and right chopsticks \n", curr);
	pthread_mutex_lock(&chopstick_lock[curr]);
	pthread_mutex_lock(&chopstick_lock[(curr + 1) % NUM_PHILOSOPHERS]);
	chopstick[curr] = 1; /* Set left chopstick status as free */
	chopstick[(curr + 1) % NUM_PHILOSOPHERS] = 1; /* Set right chopstick status as free */
	pthread_mutex_unlock(&chopstick_lock[curr]);
	pthread_mutex_unlock(&chopstick_lock[(curr + 1) % NUM_PHILOSOPHERS]);

	/* Digest food */
	int digestTime = UD(MIN_TIME, MAX_TIME);
	sleep(digestTime);
      }
      else /* Right chopstick is unavailable */
	{
	  printf("Philosopher %d: Right chopstick is unavailable \n", curr);
	  pthread_mutex_unlock(&chopstick_lock[(curr + 1) % NUM_PHILOSOPHERS]);
	  printf("Philosopher %d: Giving up left chopstick \n", curr);
	  pthread_mutex_lock(&chopstick_lock[curr]);
	  chopstick[curr] = 1; // Set left chopstick to available
	  pthread_mutex_unlock(&chopstick_lock[curr]);
	  int waitTime = UD(MIN_TIME, MAX_TIME);
	  printf("Philosopher %d: Will wait for %d seconds and try again \n", curr, waitTime);
	  sleep(waitTime);
	}
    } 
    else{ /* Left chopstick is unavailable */
      printf("Philosopher %d: Left chopstick is unavailable \n", curr);
      pthread_mutex_unlock(&chopstick_lock[curr]);
      int waitTime = UD(MIN_TIME, MAX_TIME);
      printf("Philosopher %d: Will wait for %d seconds and try again \n", curr, waitTime);
      sleep(waitTime);
    }
    sched_yield(); // Give up the processor to another thread 
  } /* While meals are still left to be consumed */
  printf("Philospher %d: All meals have been consumed \n", curr);
  pthread_exit(0);
}

/* Returns a random number between min and max */
int UD(int min, int max){
  return((int)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX))));
}















