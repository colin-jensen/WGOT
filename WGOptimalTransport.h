//
// Created by Colin Jensen on 10/15/20.
//

#ifndef WGOT_WGOPTIMALTRANSPORT_H
#define WGOT_WGOPTIMALTRANSPORT_H

#include "WGPoissonSolverQkQkRTkRect.h"

template<int dim>
class WGOptimalTransport {
public:
    WGOptimalTransport(const int degree);
    void run();

private:
    WGDarcyEquation<dim> solver;
    void compute_hessian_determinant();
};

template<int dim>
WGOptimalTransport<dim>::WGOptimalTransport(const int degree): solver(degree)
{}

template<int dim>
void WGOptimalTransport<dim>::compute_hessian_determinant()
{
//    for (const auto &cell : solver.dof_handler.active_cell_iterators())
//    {
//        std::cout << typeid(*cell).name() << std::endl;
//    }
}

template<int dim>
void WGOptimalTransport<dim>::run()
{
    solver.run();
    compute_hessian_determinant();
}



#endif //WGOT_WGOPTIMALTRANSPORT_H
