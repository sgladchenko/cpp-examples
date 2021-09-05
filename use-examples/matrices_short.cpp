#include "../Matrix.h"

int main()
{
    SG::ColVector4d colvector1 {1.0, 2.0, 3.0, 4.0};
    SG::ColVector4d colvector2 {5.0, 6.0, 7.0, 8.0};
    SG::ColVector4d colvector_res = colvector1 + colvector2;

    return 0;
}