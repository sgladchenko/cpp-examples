#include <iostream>
#include <complex>

#include "../Matrix.h"
#include "../Timer.h"

int main()
{
    const std::complex<double> iu(0.0, 1.0);

    __SG_CONSTEXPR_MATRIX__ SG::Matrix<std::complex<double>,2,2> sigma_1 {{0.0, 1.0},  {1.0,  0.0}};
    __SG_CONSTEXPR_MATRIX__ SG::Matrix<std::complex<double>,2,2> sigma_2 {{0.0, -iu},  {iu,   0.0}};
    __SG_CONSTEXPR_MATRIX__ SG::Matrix<std::complex<double>,2,2> sigma_3 {{1.0, 0.0},  {0.0, -1.0}};

    // Test basic arithmetics

    std::cout << "sigma_1 + sigma_2:\n" << std::setw(7) << sigma_1 + sigma_2 << std::endl << std::endl;
    std::cout << "sigma_1 - sigma_2:\n" << std::setw(7) << sigma_1 - sigma_2 << std::endl << std::endl;

    std::cout << "sigma_1 * 1.0i:\n"    << std::setw(7) << sigma_1 * iu      << std::endl << std::endl;
    std::cout << "1.0i * sigma_1:\n"    << std::setw(7) << iu * sigma_1      << std::endl << std::endl;
    std::cout << "sigma_1 / 2.0:\n"     << std::setw(7) << sigma_1 / 2.0     << std::endl << std::endl;

    std::cout << "sigma_1 * sigma_2:\n" << std::setw(7) << sigma_1 * sigma_2 << std::endl << std::endl;

    // Some more sophisticated functionality

    std::cout << "norm(sigma_1): " <<  SG::norm(sigma_1) << std::endl;
    std::cout << "norm(sigma_2): " <<  SG::norm(sigma_2) << std::endl;
    std::cout << "norm(sigma_3): " <<  SG::norm(sigma_3) << std::endl << std::endl;

    // AVX2 testing

    SG::ColVector4d colvector1 {1.0, 2.0, 3.0, 4.0};
    SG::ColVector4d colvector2 {5.0, 6.0, 7.0, 8.0};

    std::cout << "colvector1 + colvector2:\n" << std::setw(3) << colvector1 + colvector2 << std::endl;
    std::cout << "colvector1 - colvector2:\n" << std::setw(3) << colvector1 - colvector2 << std::endl;
    std::cout << "colvector1 * 2.0:\n" << std::setw(3) << colvector1 * 2.0 << std::endl;
    std::cout << "colvector1 / 2.0:\n" << std::setw(3) << colvector1 / 2.0 << std::endl;

    SG::Matrix4x4d matrix4x4d1 {{ 1.0,  2.0,  3.0,  4.0},
                                { 5.0,  6.0,  7.0,  8.0},
                                { 9.0, 10.0, 11.0, 12.0},
                                {13.0, 14.0, 15.0, 16.0}};

    SG::Matrix4x4d matrix4x4d2 {{16.0, 15.0, 14.0, 13.0},
                                {12.0, 11.0, 10.0,  9.0},
                                { 8.0,  7.0,  6.0,  5.0},
                                { 4.0,  3.0,  2.0,  1.0}};

    SG::Matrix<double,4,4> matrix4x4d1_NO_AVX2 {{ 1.0,  2.0,  3.0,  4.0},
                                                { 5.0,  6.0,  7.0,  8.0},
                                                { 9.0, 10.0, 11.0, 12.0},
                                                {13.0, 14.0, 15.0, 16.0}};

    SG::Matrix<double,4,4> matrix4x4d2_NO_AVX2 {{16.0, 15.0, 14.0, 13.0},
                                                {12.0, 11.0, 10.0,  9.0},
                                                { 8.0,  7.0,  6.0,  5.0},
                                                { 4.0,  3.0,  2.0,  1.0}};

    std::cout << "matrix4x4d1 + matrix4x4d2:\n" << std::setw(3) << matrix4x4d1 + matrix4x4d2 << std::endl;
    std::cout << "matrix4x4d1 - matrix4x4d2:\n" << std::setw(3) << matrix4x4d1 - matrix4x4d2 << std::endl;
    std::cout << "matrix4x4d1 * 2.0:\n" << std::setw(3) << matrix4x4d1 * 2.0 << std::endl;
    std::cout << "matrix4x4d1 / 2.0:\n" << std::setw(3) << matrix4x4d1 / 2.0 << std::endl;

    std::cout << "matrix4x4d1 * matrix4x4d2:\n" << std::setw(3) << matrix4x4d1 *matrix4x4d2 << std::endl;

    SG::Timer<std::chrono::milliseconds, std::chrono::high_resolution_clock> clock;

    /*
    clock.tick();
    for (std::size_t i = 0; i < 1000000000; ++i) { matrix4x4d1 + matrix4x4d2; }
    clock.tock();
    std::cout << "Elapsed time (matrix + matrix) (in ms): " << clock.duration().count() << std::endl;

    clock.tick();
    for (std::size_t i = 0; i < 1000000000; ++i) { matrix4x4d1 * 2.0; }
    clock.tock();
    std::cout << "Elapsed time (matrix * number) (in ms): " << clock.duration().count() << std::endl;

    clock.tick();
    for (std::size_t i = 0; i < 1000000000; ++i) { colvector1 + colvector2; }
    clock.tock();
    std::cout << "Elapsed time (vector + vector) (in ms): " << clock.duration().count() << std::endl;
    */

    clock.tick();
    for (std::size_t i = 0; i < 100000000; ++i) { matrix4x4d1 * matrix4x4d2; }
    clock.tock();
    std::cout << "Elapsed time (matrix4x41 * matrix4x42, AVX2) (in ms): " << clock.duration().count() << std::endl;

    clock.tick();
    for (std::size_t i = 0; i < 100000000; ++i) { matrix4x4d1_NO_AVX2 * matrix4x4d2_NO_AVX2; }
    clock.tock();
    std::cout << "Elapsed time (matrix4x41 * matrix4x42, No AVX2) (in ms): " << clock.duration().count() << std::endl;

    return 0;
}