#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/table_handler.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/******************************************************************************
 * Main class of the program. The purpose is to solve a simplified version of
 * the Monge-Ampere equation with Weak Galerkin methodology. The program
 * works iteratively by solving a Poisson problem with homogeneous Neumann
 * boundary conditions. On each iteration, a discrete weak hessian determinant
 * is computed and integrated into the system RHS. The program currently
 * iterates a fixed amount of times.
******************************************************************************/
template <int dim>
class WGOptimalTransport
{
public:
    WGOptimalTransport(const unsigned int degree);
    void run();

private:
    void make_grid(unsigned int n_refinements);
    void setup_system();
    void assemble_system_matrix();
    void assemble_system_rhs();
    void compute_hessian_det();
    double compute_solution_error();
    void solve();
    void output_results() const;

    Triangulation<dim> triangulation;

    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;

    FE_DGRaviartThomas<dim> fe_dgrt;
    DoFHandler<dim>         dof_handler_dgrt;
    Vector<double>          darcy_velocity;

    FullMatrix<double> hessian_det;

    TableHandler output_table;
};

/******************************************************************************
 * Class to contain information about the system RHS. Theoretically, this
 * should be a discrete probability distribution. Practically, the program
 * has been tested with smooth functions will sufficiently small values.
******************************************************************************/
template <int dim>
class RightHandSide : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};



template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    if (dim == 2) {
        // Info for sin^2(pi x) * sin^2(pi y)
        double dxx = 2.0 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[0]) *
                     std::pow(std::sin(numbers::PI * p[1]), 2);
        double dxy = std::pow(numbers::PI, 2) * std::sin(2 * numbers::PI * p[0]) *
                     std::sin(2 * numbers::PI * p[1]);
        double dyy = 2.0 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[1]) *
                     std::pow(std::sin(numbers::PI * p[0]), 2);

        // Info for cos(pi x) * cos(pi y)
//        double dxx = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
//                             std::cos(numbers::PI * p[1]);
//        double dxy = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
//                             std::sin(numbers::PI * p[1]);
//        double dyy = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
//                             std::cos(numbers::PI * p[1]);

        double laplacian = 0.1 * (dxx + dyy);
        double hessian_det = std::pow(0.1, 2) * (dxx * dyy - dxy * dxy);

        return 1.0 + laplacian + hessian_det;
//        return laplacian;
    }
    else if (dim == 3) {
        FullMatrix<double> hess(3, 3);

        // Info for sin^2(pi x) * sin^2(pi y) * sin^2(pi z)
//        hess[0][0] = 2.0 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[0]) *
//                     std::pow(std::sin(numbers::PI * p[1]), 2) *
//                     std::pow(std::sin(numbers::PI * p[2]), 2);
//        hess[1][1] = 2.0 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[1]) *
//                     std::pow(std::sin(numbers::PI * p[0]), 2) *
//                     std::pow(std::sin(numbers::PI * p[2]), 2);
//        hess[2][2] = 2.0 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[2]) *
//                     std::pow(std::sin(numbers::PI * p[0]), 2) *
//                     std::pow(std::sin(numbers::PI * p[1]), 2);
//
//        hess[0][1] = std::pow(numbers::PI, 2) * std::sin(2 * numbers::PI * p[0]) *
//                     std::sin(2 * numbers::PI * p[1]) *
//                     std::pow(std::sin(numbers::PI * p[2]), 2);
//        hess[0][2] = std::pow(numbers::PI, 2) * std::sin(2 * numbers::PI * p[0]) *
//                     std::sin(2 * numbers::PI * p[2]) *
//                     std::pow(std::sin(numbers::PI * p[1]), 2);
//        hess[1][2] = std::pow(numbers::PI, 2) * std::sin(2 * numbers::PI * p[1]) *
//                     std::sin(2 * numbers::PI * p[2]) *
//                     std::pow(std::sin(numbers::PI * p[0]), 2);

        // Info for cos(pi x) * cos(pi y) * cos(pi z)
        hess[0][0] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                     std::cos(numbers::PI * p[1]) *
                     std::cos(numbers::PI * p[2]);
        hess[1][1] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                     std::cos(numbers::PI * p[1]) *
                     std::cos(numbers::PI * p[2]);
        hess[2][2] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                     std::cos(numbers::PI * p[1]) *
                     std::cos(numbers::PI * p[2]);

        hess[0][1] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                     std::sin(numbers::PI * p[1]) *
                     std::cos(numbers::PI * p[2]);
        hess[0][2] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                     std::sin(numbers::PI * p[2]) *
                     std::cos(numbers::PI * p[1]);
        hess[1][2] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[1]) *
                     std::sin(numbers::PI * p[2]) *
                     std::cos(numbers::PI * p[0]);

        hess[1][0] = hess[0][1];
        hess[2][0] = hess[0][2];
        hess[2][1] = hess[1][2];

        double laplacian = 0.1 * (hess[0][0] + hess[1][1] + hess[2][2]);
        double dH = std::pow(0.1, 3) * hess.determinant();


        return 1.0 + laplacian + dH;
    }
}

/******************************************************************************
 * A smooth, small-valued test function.
******************************************************************************/
template <int dim>
class Cos_pi_x_Cos_pi_y : public Function<dim>
{
public:
    Cos_pi_x_Cos_pi_y()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual SymmetricTensor<2, dim> hessian(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                              const unsigned int component) const override;
    double hessian_det(const Point<dim> & p);
};

template <int dim>
double Cos_pi_x_Cos_pi_y<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    double return_value = 1.0;
    for (unsigned int d = 0; d < dim; ++d) {
        return_value *= std::cos(numbers::PI * p[d]);
    }
    return 0.1 * return_value;
}

template <int dim>
Tensor<1, dim> Cos_pi_x_Cos_pi_y<dim>::gradient(const Point<dim> & p,
                                  const unsigned int /*component*/) const
{
    Tensor<1, dim> grad;
    grad[0] = -numbers::PI * std::sin(numbers::PI * p[0]) * std::cos(numbers::PI * p[1]);
    grad[1] = -numbers::PI * std::sin(numbers::PI * p[1]) * std::cos(numbers::PI * p[0]);

    return 0.1 * grad;
}

template <int dim>
SymmetricTensor<2, dim> Cos_pi_x_Cos_pi_y<dim>::hessian(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    SymmetricTensor<2, dim> return_value;
    return_value[0][0] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                         std::cos(numbers::PI * p[1]);
    return_value[0][1] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                         std::sin(numbers::PI * p[1]);
    return_value[1][0] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                         std::sin(numbers::PI * p[1]);
    return_value[1][1] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                         std::cos(numbers::PI * p[1]);
    return 0.1 * return_value;
}

template <int dim>
double Cos_pi_x_Cos_pi_y<dim>::hessian_det(const Point<dim> & p)
{
    FullMatrix<double> hess(dim, dim);

    hess[0][0] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                 std::cos(numbers::PI * p[1]) *
                 std::cos(numbers::PI * p[2]);
    hess[1][1] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                 std::cos(numbers::PI * p[1]) *
                 std::cos(numbers::PI * p[2]);
    hess[2][2] = -numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                 std::cos(numbers::PI * p[1]) *
                 std::cos(numbers::PI * p[2]);

    hess[0][1] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                 std::sin(numbers::PI * p[1]) *
                 std::cos(numbers::PI * p[2]);
    hess[0][2] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                 std::sin(numbers::PI * p[2]) *
                 std::cos(numbers::PI * p[1]);
    hess[1][2] = numbers::PI * numbers::PI * std::sin(numbers::PI * p[1]) *
                 std::sin(numbers::PI * p[2]) *
                 std::cos(numbers::PI * p[0]);

    hess[1][0] = hess[0][1];
    hess[2][0] = hess[0][2];
    hess[2][1] = hess[1][2];

    double laplacian = 0.1 * (hess[0][0] + hess[1][1] + hess[2][2]);
    double dH = std::pow(0.1, 3) * hess.determinant();

    return dH;
}

/******************************************************************************
 * A smooth, small-valued test function.
******************************************************************************/
template <int dim>
class Sin_2_pi_x_Sin_2_pi_y : public Function<dim>
{
public:
    Sin_2_pi_x_Sin_2_pi_y()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual SymmetricTensor<2, dim> hessian(const Point<dim> & p,
                                              const unsigned int component) const override;
    virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                    const unsigned int component) const override;
    double hessian_det(const Point<dim> & p);
};



template <int dim>
double Sin_2_pi_x_Sin_2_pi_y<dim>::value(const Point<dim> &p,
                                         const unsigned int /*component*/) const
{
    double return_value = 1.0;
    for (unsigned int d = 0; d < dim; ++d) {
        return_value *= std::pow(std::sin(numbers::PI * p[d]), 2);
    }
    return 0.1 * return_value;
}

template <int dim>
Tensor<1, dim> Sin_2_pi_x_Sin_2_pi_y<dim>::gradient(const Point<dim> & p,
                                                    const unsigned int /*component*/) const
{
    Tensor<1, dim> grad;
    grad[0] = numbers::PI * std::sin(2 * numbers::PI * p[0]) *
                            std::sin(numbers::PI * p[1]) *
                            std::sin(numbers::PI * p[1]);
    grad[1] = numbers::PI * std::sin(2 * numbers::PI * p[1]) *
                            std::sin(numbers::PI * p[1]) *
                            std::sin(numbers::PI * p[0]);

    return grad;
}

template <int dim>
SymmetricTensor<2, dim> Sin_2_pi_x_Sin_2_pi_y<dim>::hessian(const Point<dim> &p,
                                                              const unsigned int /*component*/) const
{
    SymmetricTensor<2, dim> return_value;
    return_value[0][0] = 2 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[0]) *
                                                        std::pow(std::sin(numbers::PI * p[1]), 2);
    return_value[0][1] = std::pow(numbers::PI, 2) * std::sin(2 * numbers::PI * p[0]) *
                                                    std::sin(2 * numbers::PI * p[1]);
    return_value[1][1] = 2 * std::pow(numbers::PI, 2) * std::cos(2 * numbers::PI * p[1]) *
                                                        std::pow(std::sin(numbers::PI * p[0]), 2);
    return 0.05 * return_value;
}

template<int dim>
double Sin_2_pi_x_Sin_2_pi_y<dim>::hessian_det(const Point<dim> &p) {
    auto hess = hessian(p, 0);
    return hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
}

// In this constructor, we create a finite element space for vector valued
// functions, which will here include the ones used for the interior and
// interface values, $p^\circ$ and $p^\partial$.
template <int dim>
WGOptimalTransport<dim>::WGOptimalTransport(const unsigned int degree)
        : fe(FE_DGQ<dim>(degree), 1, FE_FaceQ<dim>(degree), 1)
        , dof_handler(triangulation)
        , fe_dgrt(degree)
        , dof_handler_dgrt(triangulation)
{}

// We generate a mesh on the unit square domain and refine it.
template <int dim>
void WGOptimalTransport<dim>::make_grid(unsigned int n_refinements)
{
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(n_refinements);
}

// Link DOFHandlers with finite elements and handle the uniqueness issue
// resulting from the Neumann boundary conditions. The method used here
// is forcing boundary values to have a given mean value.
template <int dim>
void WGOptimalTransport<dim>::setup_system()
{
    // Setup DOFHandlers
    dof_handler.distribute_dofs(fe);
    dof_handler_dgrt.distribute_dofs(fe_dgrt);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Handle uniqueness problem that results from Neumann boundary conditions
    const FEValuesExtractors::Scalar interface_pressure(1);
    const ComponentMask              interface_pressure_mask =
        fe.component_mask(interface_pressure);
    std::vector<bool> boundary_dofs(dof_handler.n_dofs(), false);
    DoFTools::extract_boundary_dofs(dof_handler,
                                    interface_pressure_mask,
                                    boundary_dofs);
    const unsigned int first_boundary_dof = std::distance(
            boundary_dofs.begin(),
            std::find(boundary_dofs.begin(), boundary_dofs.end(), true));
    constraints.clear();
    constraints.add_line(first_boundary_dof);
    for (unsigned int i = first_boundary_dof + 1; i < dof_handler.n_dofs(); ++i)
        if (boundary_dofs[i] == true)
            constraints.add_entry(first_boundary_dof, i, -1);
    constraints.close();

    // Setup system matrix
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

// Build the system matrix according to weak Galerkin methodology
template <int dim>
void WGOptimalTransport<dim>::assemble_system_matrix()
{
    const QGauss<dim>     quadrature_formula(fe_dgrt.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe_dgrt.degree + 1);

    FEValues<dim>     fe_values(fe,
                                quadrature_formula,
                                update_values | update_quadrature_points |
                                update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

    FEValues<dim>     fe_values_dgrt(fe_dgrt,
                                     quadrature_formula,
                                     update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);
    FEFaceValues<dim> fe_face_values_dgrt(fe_dgrt,
                                          face_quadrature_formula,
                                          update_values |
                                          update_normal_vectors |
                                          update_quadrature_points |
                                          update_JxW_values);

    const unsigned int dofs_per_cell      = fe.dofs_per_cell;
    const unsigned int dofs_per_cell_dgrt = fe_dgrt.dofs_per_cell;

    const unsigned int n_q_points      = fe_values.get_quadrature().size();
    const unsigned int n_q_points_dgrt = fe_values_dgrt.get_quadrature().size();

    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

    RightHandSide<dim>  right_hand_side;
    std::vector<double> right_hand_side_values(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Next, let us declare the various cell matrices discussed in the
    // introduction:
    FullMatrix<double> cell_matrix_M(dofs_per_cell_dgrt, dofs_per_cell_dgrt);
    FullMatrix<double> cell_matrix_G(dofs_per_cell_dgrt, dofs_per_cell);
    FullMatrix<double> cell_matrix_C(dofs_per_cell, dofs_per_cell_dgrt);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    Vector<double>     cell_solution(dofs_per_cell);

    // We need <code>FEValuesExtractors</code> to access the @p interior and
    // @p face component of the shape functions.
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_interior(0);
    const FEValuesExtractors::Scalar pressure_face(1);

    // This finally gets us in position to loop over all cells. On
    // each cell, we will first calculate the various cell matrices
    // used to construct the local matrix -- as they depend on the
    // cell in question, they need to be re-computed on each cell. We
    // need shape functions for the Raviart-Thomas space as well, for
    // which we need to create first an iterator to the cell of the
    // triangulation, which we can obtain by assignment from the cell
    // pointing into the DoFHandler.
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        const typename Triangulation<dim>::active_cell_iterator cell_dgrt =
                cell;
        fe_values_dgrt.reinit(cell_dgrt);

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   right_hand_side_values);

        // The first cell matrix we will compute is the mass matrix
        // for the Raviart-Thomas space.  Hence, we need to loop over
        // all the quadrature points for the velocity FEValues object.
        cell_matrix_M = 0;
        for (unsigned int q = 0; q < n_q_points_dgrt; ++q)
            for (unsigned int i = 0; i < dofs_per_cell_dgrt; ++i)
            {
                const Tensor<1, dim> v_i = fe_values_dgrt[velocities].value(i, q);
                for (unsigned int k = 0; k < dofs_per_cell_dgrt; ++k)
                {
                    const Tensor<1, dim> v_k =
                            fe_values_dgrt[velocities].value(k, q);
                    cell_matrix_M(i, k) += (v_i * v_k * fe_values_dgrt.JxW(q));
                }
            }
        // Next we take the inverse of this matrix by using
        // FullMatrix::gauss_jordan(). It will be used to calculate
        // the coefficient matrix $C^K$ later. It is worth recalling
        // later that `cell_matrix_M` actually contains the *inverse*
        // of $M^K$ after this call.
        cell_matrix_M.gauss_jordan();

        // From the introduction, we know that the right hand side
        // $G^K$ of the equation that defines $C^K$ is the difference
        // between a face integral and a cell integral. Here, we
        // approximate the negative of the contribution in the
        // interior. Each component of this matrix is the integral of
        // a product between a basis function of the polynomial space
        // and the divergence of a basis function of the
        // Raviart-Thomas space. These basis functions are defined in
        // the interior.
        cell_matrix_G = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell_dgrt; ++i)
            {
                const double div_v_i =
                        fe_values_dgrt[velocities].divergence(i, q);
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j_interior =
                            fe_values[pressure_interior].value(j, q);

                    cell_matrix_G(i, j) -=
                            (div_v_i * phi_j_interior * fe_values.JxW(q));
                }
            }


        // Next, we approximate the integral on faces by quadrature.
        // Each component is the integral of a product between a basis function
        // of the polynomial space and the dot product of a basis function of
        // the Raviart-Thomas space and the normal vector. So we loop over all
        // the faces of the element and obtain the normal vector.
        for (const auto &face : cell->face_iterators())
        {
            fe_face_values.reinit(cell, face);
            fe_face_values_dgrt.reinit(cell_dgrt, face);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
                const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                for (unsigned int i = 0; i < dofs_per_cell_dgrt; ++i)
                {
                    const Tensor<1, dim> v_i =
                            fe_face_values_dgrt[velocities].value(i, q);
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const double phi_j_face =
                                fe_face_values[pressure_face].value(j, q);

                        cell_matrix_G(i, j) +=
                                ((v_i * normal) * phi_j_face * fe_face_values.JxW(q));
                    }
                }
            }
        }

        // @p cell_matrix_C is then the matrix product between the
        // transpose of $G^K$ and the inverse of the mass matrix
        // (where this inverse is stored in @p cell_matrix_M):
        cell_matrix_G.Tmmult(cell_matrix_C, cell_matrix_M);

        // Finally we can compute the local matrix $A^K$.  Element
        // $A^K_{ij}$ is given by $\int_{E} \sum_{k,l} C_{ik} C_{jl}
        // (\mathbf{K} \mathbf{v}_k) \cdot \mathbf{v}_l
        // \mathrm{d}x$. We have calculated the coefficients $C$ in
        // the previous step, and so obtain the following after
        // suitably re-arranging the loops:
        local_matrix = 0;
        for (unsigned int q = 0; q < n_q_points_dgrt; ++q)
        {
            for (unsigned int k = 0; k < dofs_per_cell_dgrt; ++k)
            {
                const Tensor<1, dim> v_k =
                        fe_values_dgrt[velocities].value(k, q);
                for (unsigned int l = 0; l < dofs_per_cell_dgrt; ++l)
                {
                    const Tensor<1, dim> v_l =
                            fe_values_dgrt[velocities].value(l, q);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            local_matrix(i, j) +=
                                    (cell_matrix_C[i][k] * v_k) *
                                    cell_matrix_C[j][l] * v_l * fe_values_dgrt.JxW(q);
                }
            }
        }

        // The last step is to distribute components of the local
        // matrix into the system matrix and transfer components of
        // the cell right hand side into the system right hand side:
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                local_matrix, local_dof_indices, system_matrix);
    }
}

// Build the system RHS (with the hessian determinant being incorporated)
template<int dim>
void WGOptimalTransport<dim>::assemble_system_rhs()
{
    const QGauss<dim>     quadrature_formula(fe_dgrt.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe_dgrt.degree + 1);

    FEValues<dim>     fe_values(fe,
                                quadrature_formula,
                                update_values | update_quadrature_points | update_gradients |
                                update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);


    const unsigned int dofs_per_cell      = fe.dofs_per_cell;

    const unsigned int n_q_points      = fe_values.get_quadrature().size();

    RightHandSide<dim>  right_hand_side;
    std::vector<double> right_hand_side_values(n_q_points);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // We need <code>FEValuesExtractors</code> to access the @p interior and
    // @p face component of the shape functions.
    //const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_interior(0);
    const FEValuesExtractors::Scalar pressure_face(1);

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();

    for (; cell != endc; ++cell)
    {
        fe_values.reinit(cell);

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   right_hand_side_values);

        cell_rhs = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += fe_values[pressure_interior].value(i, q) *
                        (1.0 - right_hand_side_values[q] + hessian_det[cell->active_cell_index()][q]) *
                        fe_values.JxW(q);
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                cell_rhs, local_dof_indices, system_rhs);
    }
}

// Solve the global finite element system
template <int dim>
void WGOptimalTransport<dim>::solve()
{
    SolverControl            solver_control(10000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    constraints.distribute(solution);
}

// Compute the discrete weak hessian determinant of the solution function from
// the previous iteration.
template <int dim>
void WGOptimalTransport<dim>::compute_hessian_det()
{
    // A finite element acting as basis functions to the discrete weak 2nd order partial derivative
    FE_DGQ<dim> fe_h(0);
    DoFHandler<dim> dof_handler_h(triangulation);
    dof_handler_h.distribute_dofs(fe_h);

    // A finite element to handle the values of the function we are taking the
    // discrete weak 2nd order partial derivative of
    FE_DGQ<dim> fe_f(fe.degree);
    DoFHandler<dim> dof_handler_f(triangulation);
    dof_handler_f.distribute_dofs(fe_f);

    const QGauss<dim>     quadrature_formula(fe_dgrt.degree + 1);
    const QGauss<dim - 1> quadrature_formula_face(fe_dgrt.degree + 1);

    // For accessing the values of the DW2PD basis functions on cell interiors
    FEValues<dim> fe_values_h(fe_h,
                              quadrature_formula,
                              update_values |
                              update_hessians |
                              update_quadrature_points |
                              update_JxW_values);

    // For accessing the values of the DW2PD basis functions on cell faces
    FEFaceValues<dim> fe_values_h_face(fe_h,
                                       quadrature_formula_face,
                                       update_values |
                                       update_gradients |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // For accessing the values of the function to be differentiated on the cell interior
    FEValues<dim> fe_values_f(fe_f,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points);

    // For accessing the values of the function to be differentiated on the cell faces
    FEFaceValues<dim> fe_values_f_face(fe_f,
                                       quadrature_formula_face,
                                       update_values |
                                       update_gradients |
                                       update_quadrature_points);

    const unsigned int n_quad_points = quadrature_formula.size();
    const unsigned int n_quad_points_face = quadrature_formula_face.size();
    const unsigned int dofs_per_cell_h = fe_values_h.dofs_per_cell;

    // Make sure the list of hessian determinants is of the right length
    hessian_det.reinit(triangulation.n_active_cells(), n_quad_points);

    FullMatrix<double> cell_matrix_M(dofs_per_cell_h, dofs_per_cell_h);
    Vector<double> cell_vector_G(dofs_per_cell_h);
    Vector<double> cell_dw2pd_coeffs(dofs_per_cell_h);

    /***************************************************************************
     ***************************************************************************/

    // Setup cell iterators
    typename DoFHandler<dim>::active_cell_iterator
        cell   = dof_handler.begin_active(),
        cell_f = dof_handler_f.begin_active(),
        cell_h = dof_handler_h.begin_active(),
        endc   = dof_handler_f.end();

    // Build mass matrix, which will be the same for every cell
    fe_values_h.reinit(cell_h);
    cell_matrix_M = 0;
    for (unsigned int q = 0; q < n_quad_points; ++q) {
        for (unsigned int k = 0; k < dofs_per_cell_h; ++k) {
            for (unsigned int m = 0; m < dofs_per_cell_h; ++m) {
                cell_matrix_M(k, m) += fe_values_h.shape_value(k, q) *
                                       fe_values_h.shape_value(m, q) *
                                       fe_values_h.JxW(q);
            }
        }
    }
    cell_matrix_M.gauss_jordan(); // From here, cell_matrix_M will contain its inverse

    // Start iterating over cell
    for (; cell_f != endc; ++cell, ++cell_f, ++cell_h) {
        fe_values_f.reinit(cell_f);
        fe_values_h.reinit(cell_h);

        std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
        std::vector<types::global_dof_index> interior_dof_indices;

        cell->get_dof_indices(local_dof_indices);

        // Filter out the dofs pertaining to the interior component of the FESystem
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
            auto index_pair = fe.system_to_component_index(i);
            if (index_pair.first == 0) {
                interior_dof_indices.push_back(local_dof_indices[i]);
            }
        }

        // Storage container for the discrete weak second partial derivative
        // on this cell
        std::vector<FullMatrix<double>> cell_hessian(n_quad_points);
        for (unsigned int q = 0; q < n_quad_points; ++q) {
            cell_hessian[q].reinit(dim, dim);
        }

        // Loop over each spatial dimension
        for (unsigned int d1 = 0; d1 < dim; ++d1) {
            for (unsigned int d2 = 0; d2 < dim; ++d2) {

                cell_vector_G = 0;
                cell_dw2pd_coeffs = 0;

                // TODO: Incorporate interior part of G (later)

                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f) {

                    fe_values_f_face.reinit(cell_f, cell_f->face(f));
                    fe_values_h_face.reinit(cell_h, cell_h->face(f));

                    // TODO: Incorporate face part of G (later)


                    // Get values of the function gradient along the current face
                    std::vector<Tensor<1, dim>> function_gradients(n_quad_points_face);
                    fe_values_f_face.get_function_gradients(solution, interior_dof_indices, function_gradients);

                    // Add the contribution of this face onto the RHS of the integral equation
                    // which approximated the discrete weak second partial derivative
                    for (unsigned int q = 0; q < n_quad_points_face; ++q) {

                        const auto normal = fe_values_h_face.normal_vector(q);
                        for (unsigned int k = 0; k < dofs_per_cell_h; ++k) {
                            cell_vector_G(k) += function_gradients[q][d1] *
                                                normal[d2] *
                                                fe_values_h_face.shape_value(k, q) *
                                                fe_values_h_face.JxW(q);
                        }
                    }
                }

                // Having compiled the contribution of each face to the RHS,
                // now compute the discrete weak second partial derivated
                // in the current spacial dimensions at each quadrature point
                cell_matrix_M.vmult(cell_dw2pd_coeffs, cell_vector_G);
                std::vector<double> cell_dw2pds(n_quad_points);
                std::vector<types::global_dof_index> dof_indices_h = {0};
                fe_values_h.get_function_values(cell_dw2pd_coeffs, dof_indices_h, cell_dw2pds);

                for (unsigned int q = 0; q < n_quad_points; ++q) {
                    cell_hessian[q](d1, d2) = cell_dw2pds[q];
                }
            }
        }

        // Take the discrete weak second partial derivatives computed for this cell and
        // use them to compute the discrete weak hessian determinants.
        for (unsigned int q = 0; q < n_quad_points; ++q) {
            hessian_det[cell->active_cell_index()][q] = cell_hessian[q].determinant();
        }
    }
}

// Compute the L2 error in the solution as compared
// with the exact function.
template <int dim>
double WGOptimalTransport<dim>::compute_solution_error()
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    const ComponentSelectFunction<dim> select_interior_pressure(0, 2);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Sin_2_pi_x_Sin_2_pi_y<dim> (),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm,
                                      &select_interior_pressure);
    const double L2_error = difference_per_cell.l2_norm();
    std::cout << "L2_error_pressure " << L2_error << std::endl;

    return L2_error;
}

// Call the class functions, run the fixed-point iteration
template <int dim>
void WGOptimalTransport<dim>::run()
{
    make_grid(3);
    setup_system();
    assemble_system_matrix();

    solution = 0;

    for (unsigned int i = 0; i < 10; ++i) {
        compute_hessian_det();
        assemble_system_rhs();
        solve();
        double err = compute_solution_error();
        if (err > 1.) break;
    }

}

int main()
{
    try
    {
        WGOptimalTransport<2> wgot(2);
        wgot.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}