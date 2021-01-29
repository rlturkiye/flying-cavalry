#include "Vector.hpp"

int main () {
  Vector a(1,2), b(2, 1);
  a = a + b;
  a.print();
  return 0;
}