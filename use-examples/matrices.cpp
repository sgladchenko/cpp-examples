#include <iostream>
#include <complex>

#include "../Matrix.h"

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
    std::cout << "norm(sigma_3): " <<  SG::norm(sigma_3) << std::endl;

    return 0;
}