#include <stdio.h>
#include <stdlib.h>

int main() {
  double start = -5;
  double step = 1.4;
  int64_t num_steps = 8;

  printf("step: %.20e\n", step);
  printf("start: %.20e\n", start);

  for (int64_t ind = 0; ind < num_steps; ind++) {
    printf("%ld: %.20e\n", ind, (start + step * ind));
  }
}
