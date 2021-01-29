#ifndef PROJECT_VECTOR_
  #define PROJECT_VECTOR_

class Vector {
  double x,y;
public:
  Vector();
  Vector(double,double);
  Vector operator+(const Vector &other) const;
  void print();
};

#endif