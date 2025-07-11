#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" double compute_gold(float, float, int, float);

/*------------------------------------------------------------------
 *  * Function:    f
 *   * Purpose:     Compute value of function to be integrated
 *    * Input args:  x
 *     * Output: sqrt((1 + x*x)/(1 + x*x*x*x))
 *      */

float f(float x)
{
    return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/*------------------------------------------------------------------
 *  * Function:    Trap
 *   * Purpose:     Estimate integral from a to b of f using trap rule and
 *    *              n trapezoids
 *     * Input args:  a, b, n, h
 *      * Return val:  Estimate of the integral
 *       */

double compute_gold(float a, float b, int n, float h)
{
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a + k*h);
   }

   integral = integral*h;
   return integral;
}
