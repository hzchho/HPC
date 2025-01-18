#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

int min(int a, int b);
void parallel_for(int start, int end, int increment, void*(functor)(void*), void *arg, int num_threads);

#endif