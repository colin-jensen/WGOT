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

#include <fstream>
#include <iostream>




using namespace dealii;

// @sect3{The WGOptimalTransport class template}

// This is the main class of this program. We will solve for the numerical
// pressure in the interior and on faces using the weak Galerkin (WG) method,
// and calculate the $L_2$ error of pressure. In the post-processing step, we
// will also calculate $L_2$-errors of the velocity and flux.
//
// The structure of the class is not fundamentally different from that of
// previous tutorial programs, so there is little need to comment on the
// details with one exception: The class has a member variable `fe_dgrt`
// that corresponds to the "broken" Raviart-Thomas space mentioned in the
// introduction. There is a matching `dof_handler_dgrt` that represents a
// global enumeration of a finite element field created from this element, and
// a vector `darcy_velocity` that holds nodal values for this field. We will
// use these three variables after solving for the pressure to compute a
// postprocessed velocity field for which we can then evaluate the error
// and which we can output for visualization.
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
    void compute_hessian();
    void compute_pressure_error();
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

    Vector<double> hessian_det;
};



// @sect3{Right hand side, boundary values, and exact solution}

// Next, we define the coefficient matrix $\mathbf{K}$ (here, the
// identity matrix), Dirichlet boundary conditions, the right-hand
// side $f = 2\pi^2 \sin(\pi x) \sin(\pi y)$, and the exact solution
// that corresponds to these choices for $K$ and $f$, namely $p =
// \sin(\pi x) \sin(\pi y)$.
template <int dim>
class Coefficient : public TensorFunction<2, dim>
{
public:
    Coefficient()
            : TensorFunction<2, dim>()
    {}

    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<Tensor<2, dim>> &values) const override;
};



template <int dim>
void Coefficient<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<2, dim>> &  values) const
{
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));
    for (unsigned int p = 0; p < points.size(); ++p)
        values[p] = unit_symmetric_tensor<dim>();
}



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    BoundaryValues()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
};



template <int dim>
double BoundaryValues<dim>::value(const Point<dim> & p,
                                  const unsigned int /*component*/) const
{
    return std::cos(numbers::PI * p[0]) * std::cos(numbers::PI * p[1]);
}



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
    double laplacian = 0.1 * -2 * numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                                                              std::cos(numbers::PI * p[1]);
    double hessian_determinant = 0.1 * 0.1 * 0.5 * std::pow(numbers::PI, 4) * (std::cos(2 * numbers::PI * p[0]) +
                                                                               std::cos(2 * numbers::PI * p[1]));
//    return  laplacian + hessian_determinant + 1;
    return laplacian;
}



// The class that implements the exact pressure solution has an
// oddity in that we implement it as a vector-valued one with two
// components. (We say that it has two components in the constructor
// where we call the constructor of the base Function class.) In the
// `value()` function, we do not test for the value of the
// `component` argument, which implies that we return the same value
// for both components of the vector-valued function. We do this
// because we describe the finite element in use in this program as
// a vector-valued system that contains the interior and the
// interface pressures, and when we compute errors, we will want to
// use the same pressure solution to test both of these components.
template <int dim>
class Cos_pi_x_Cos_pi_y : public Function<dim>
{
public:
    Cos_pi_x_Cos_pi_y()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual SymmetricTensor<dim, dim> hessian(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                              const unsigned int component) const override;
    double hessian_det(const Point<dim> & p);
};



template <int dim>
double Cos_pi_x_Cos_pi_y<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    return 0.1 * std::cos(numbers::PI * p[0]) * std::cos(numbers::PI * p[1]);
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
SymmetricTensor<dim, dim> Cos_pi_x_Cos_pi_y<dim>::hessian(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    SymmetricTensor<dim, dim> return_value;
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
    return 0.1 * 0.1 * 0.5 * std::pow(numbers::PI, 4) * (std::cos(2 * numbers::PI * p[0]) + std::cos(2 * numbers::PI * p[1]));
}

template <int dim>
class Sin_pi_x_Sin_pi_y : public Function<dim>
{
public:
    Sin_pi_x_Sin_pi_y()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual SymmetricTensor<dim, dim> hessian(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                    const unsigned int component) const override;
};



template <int dim>
double Sin_pi_x_Sin_pi_y<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim> Sin_pi_x_Sin_pi_y<dim>::gradient(const Point<dim> & p,
                                                const unsigned int /*component*/) const
{
    Tensor<1, dim> grad;
    grad[0] = numbers::PI * std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
    grad[1] = numbers::PI * std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[0]);

    return grad;
}

template <int dim>
SymmetricTensor<dim, dim> Sin_pi_x_Sin_pi_y<dim>::hessian(const Point<dim> &p,
                                                          const unsigned int /*component*/) const
{
    SymmetricTensor<dim, dim> return_value;
    return_value[0][0] = -numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                         std::sin(numbers::PI * p[1]);
    return_value[0][1] = numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                         std::cos(numbers::PI * p[1]);
    return_value[1][0] = numbers::PI * numbers::PI * std::cos(numbers::PI * p[0]) *
                         std::cos(numbers::PI * p[1]);
    return_value[1][1] = -numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
                         std::sin(numbers::PI * p[1]);
    return return_value;
}

template <int dim>
class X_2_Y_2 : public Function<dim>
{
public:
    X_2_Y_2()
            : Function<dim>(2)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
    virtual SymmetricTensor<dim, dim> hessian(const Point<dim> & p,
                                              const unsigned int component) const override;
};



template <int dim>
double X_2_Y_2<dim>::value(const Point<dim> &p,
                                     const unsigned int /*component*/) const
{
    return p[0] * p[0] * p[1] * p[1];
}

template <int dim>
SymmetricTensor<dim, dim> X_2_Y_2<dim>::hessian(const Point<dim> &p,
                                                          const unsigned int /*component*/) const
{
    SymmetricTensor<dim, dim> return_value;
    return_value[0][0] = 2*p[1]*p[1];
    return_value[0][1] = 4*p[0]*p[1];
    return_value[1][0] = 4*p[0]*p[1];
    return_value[1][1] = 2*p[0]*p[0];
    return return_value;
}


// @sect3{WGOptimalTransport class implementation}

// @sect4{WGOptimalTransport::WGOptimalTransport}

// In this constructor, we create a finite element space for vector valued
// functions, which will here include the ones used for the interior and
// interface pressures, $p^\circ$ and $p^\partial$.
template <int dim>
WGOptimalTransport<dim>::WGOptimalTransport(const unsigned int degree)
        : fe(FE_DGQ<dim>(degree), 1, FE_FaceQ<dim>(degree), 1)
        , dof_handler(triangulation)
        , fe_dgrt(degree)
        , dof_handler_dgrt(triangulation)
{}



// @sect4{WGOptimalTransport::make_grid}

// We generate a mesh on the unit square domain and refine it.
template <int dim>
void WGOptimalTransport<dim>::make_grid(unsigned int n_refinements)
{
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(n_refinements);

    hessian_det.reinit(triangulation.n_active_cells());

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: " << triangulation.n_cells()
              << std::endl;
}



// @sect4{WGOptimalTransport::setup_system}

// After we have created the mesh above, we distribute degrees of
// freedom and resize matrices and vectors. The only piece of
// interest in this function is how we interpolate the boundary
// values for the pressure. Since the pressure consists of interior
// and interface components, we need to make sure that we only
// interpolate onto that component of the vector-valued solution
// space that corresponds to the interface pressures (as these are
// the only ones that are defined on the boundary of the domain). We
// do this via a component mask object for only the interface
// pressures.
template <int dim>
void WGOptimalTransport<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    dof_handler_dgrt.distribute_dofs(fe_dgrt);

    std::cout << "   Number of pressure degrees of freedom per cell: "
              << fe.dofs_per_cell << std::endl;

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



//    {
//        constraints.clear();
//        const FEValuesExtractors::Scalar interface_pressure(1);
//        const ComponentMask              interface_pressure_mask =
//                fe.component_mask(interface_pressure);
//        VectorTools::interpolate_boundary_values(dof_handler,
//                                                 0,
//                                                 BoundaryValues<dim>(),
//                                                 constraints,
//                                                 interface_pressure_mask);
//        constraints.close();
//    }


    // In the bilinear form, there is no integration term over faces
    // between two neighboring cells, so we can just use
    // <code>DoFTools::make_sparsity_pattern</code> to calculate the sparse
    // matrix.
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}



// @sect4{WGOptimalTransport::assemble_system}

// This function is more interesting. As detailed in the
// introduction, the assembly of the linear system requires us to
// evaluate the weak gradient of the shape functions, which is an
// element in the Raviart-Thomas space. As a consequence, we need to
// define a Raviart-Thomas finite element object, and have FEValues
// objects that evaluate it at quadrature points. We then need to
// compute the matrix $C^K$ on every cell $K$, for which we need the
// matrices $M^K$ and $G^K$ mentioned in the introduction.
//
// A point that may not be obvious is that in all previous tutorial
// programs, we have always called FEValues::reinit() with a cell
// iterator from a DoFHandler. This is so that one can call
// functions such as FEValuesBase::get_function_values() that
// extract the values of a finite element function (represented by a
// vector of DoF values) on the quadrature points of a cell. For
// this operation to work, one needs to know which vector elements
// correspond to the degrees of freedom on a given cell -- i.e.,
// exactly the kind of information and operation provided by the
// DoFHandler class.
//
// We could create a DoFHandler object for the "broken" Raviart-Thomas space
// (using the FE_DGRT class), but we really don't want to here: At
// least in the current function, we have no need for any globally defined
// degrees of freedom associated with this broken space, but really only
// need to reference the shape functions of such a space on the current
// cell. As a consequence, we use the fact that one can call
// FEValues::reinit() also with cell iterators into Triangulation
// objects (rather than DoFHandler objects). In this case, FEValues
// can of course only provide us with information that only
// references information about cells, rather than degrees of freedom
// enumerated on these cells. So we can't use
// FEValuesBase::get_function_values(), but we can use
// FEValues::shape_value() to obtain the values of shape functions
// at quadrature points on the current cell. It is this kind of
// functionality we will make use of below. The variable that will
// give us this information about the Raviart-Thomas functions below
// is then the `fe_values_rt` (and corresponding `fe_face_values_rt`)
// object.
//
// Given this introduction, the following declarations should be
// pretty obvious:
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

    const Coefficient<dim>      coefficient;
    std::vector<Tensor<2, dim>> coefficient_values(n_q_points);

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
        coefficient.value_list(fe_values.get_quadrature_points(),
                               coefficient_values);

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
                                    (coefficient_values[q] * cell_matrix_C[i][k] * v_k) *
                                    cell_matrix_C[j][l] * v_l * fe_values_dgrt.JxW(q);
                }
            }
        }

        // Next, we calculate the right hand side, $\int_{K} f q \mathrm{d}x$:
//        cell_rhs = 0;
//        for (unsigned int q = 0; q < n_q_points; ++q)
//            for (unsigned int i = 0; i < dofs_per_cell; ++i)
//            {
//                cell_rhs(i) += (fe_values[pressure_interior].value(i, q) *
//                                right_hand_side_values[q] * fe_values.JxW(q));
//            }

        // The last step is to distribute components of the local
        // matrix into the system matrix and transfer components of
        // the cell right hand side into the system right hand side:
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                local_matrix, local_dof_indices, system_matrix);
    }
}

template<int dim>
void WGOptimalTransport<dim>::assemble_system_rhs()
{
    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

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
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

    RightHandSide<dim>  right_hand_side;
    std::vector<double> right_hand_side_values(n_q_points);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // We need <code>FEValuesExtractors</code> to access the @p interior and
    // @p face component of the shape functions.
    //const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_interior(0);
    const FEValuesExtractors::Scalar pressure_face(1);

    Cos_pi_x_Cos_pi_y<dim> cos_pi_x_cos_pi_y;
    Sin_pi_x_Sin_pi_y<dim> sin_pi_x_sin_pi_y;
    X_2_Y_2<dim> x_2_y_2;

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();

    std::vector<Point<3>> errors_in_solution_gradient;
    std::vector<double> errors_x;
    std::vector<double> errors_y;

    for (; cell != endc; ++cell)
    {
        fe_values.reinit(cell);

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   right_hand_side_values);



        errors_x.push_back(std::abs(fe_values));


        cell_rhs = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) += fe_values[pressure_interior].value(i, q) *
                        -right_hand_side_values[q] * //(1 - right_hand_side_values[q] + hessian_det[cell->active_cell_index()]) *
                        fe_values.JxW(q);
            }

//        for (const auto &face : cell->face_iterators()) {
//            if (face->at_boundary()){
//                fe_face_values.reinit(cell, face);
//
//                std::vector<Tensor<1, dim>> boundary_values(n_face_q_points);
//                cos_pi_x_cos_pi_y.gradient_list(fe_face_values.get_quadrature_points(), boundary_values);
//
//                for (unsigned int q = 0; q < n_face_q_points; ++q) {
//                    const auto normal = fe_face_values.normal_vector(q);
//                    const auto neumann_value = boundary_values[q] * normal;
//                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
//                        cell_rhs(i) += neumann_value *
//                                       fe_face_values[pressure_face].value(i, q) *
//                                       fe_face_values.JxW(q);
//                    }
//                }
//            }
//        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                cell_rhs, local_dof_indices, system_rhs);
    }
}



// @sect4{WGOptimalTransport<dim>::solve}

// This step is rather trivial and the same as in many previous
// tutorial programs:
template <int dim>
void WGOptimalTransport<dim>::solve()
{
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    constraints.distribute(solution);
}



// @sect4{WGOptimalTransport::output_results}

// We have two sets of results to output: the interior solution and
// the skeleton solution. We use <code>DataOut</code> to visualize
// interior results. The graphical output for the skeleton results
// is done by using the DataOutFaces class.
//
// In both of the output files, both the interior and the face
// variables are stored. For the interface output, the output file
// simply contains the interpolation of the interior pressures onto
// the faces, but because it is undefined which of the two interior
// pressure variables you get from the two adjacent cells, it is
// best to ignore the interior pressure in the interface output
// file. Conversely, for the cell interior output file, it is of
// course impossible to show any interface pressures $p^\partial$,
// because these are only available on interfaces and not cell
// interiors. Consequently, you will see them shown as an invalid
// value (such as an infinity).
//
// For the cell interior output, we also want to output the velocity
// variables. This is a bit tricky since it lives on the same mesh
// but uses a different DoFHandler object (the pressure variables live
// on the `dof_handler` object, the Darcy velocity on the `dof_handler_dgrt`
// object). Fortunately, there are variations of the
// DataOut::add_data_vector() function that allow specifying which
// DoFHandler a vector corresponds to, and consequently we can visualize
// the data from both DoFHandler objects within the same file.
template <int dim>
void WGOptimalTransport<dim>::output_results() const
{
    {
        DataOut<dim> data_out;

        // First attach the pressure solution to the DataOut object:
        const std::vector<std::string> solution_names = {"interior_pressure",
                                                         "interface_pressure"};
        data_out.add_data_vector(dof_handler, solution, solution_names);

        // Then do the same with the Darcy velocity field, and continue
        // with writing everything out into a file.
//        const std::vector<std::string> velocity_names(dim, "velocity");
//        const std::vector<
//                DataComponentInterpretation::DataComponentInterpretation>
//                velocity_component_interpretation(
//                dim, DataComponentInterpretation::component_is_part_of_vector);
//        data_out.add_data_vector(dof_handler_dgrt,
//                                 darcy_velocity,
//                                 velocity_names,
//                                 velocity_component_interpretation);

        data_out.build_patches(fe.degree);
        std::ofstream output("solution_interior.eps");
        data_out.write_eps(output);
    }

    {
        DataOutFaces<dim> data_out_faces(false);
        data_out_faces.attach_dof_handler(dof_handler);
        data_out_faces.add_data_vector(solution, "Pressure_Face");
        data_out_faces.build_patches(fe.degree);
        std::ofstream face_output("solution_interface.vtu");
        data_out_faces.write_vtu(face_output);
    }
}

template <int dim>
void WGOptimalTransport<dim>::compute_hessian()
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

    Point<dim> cell_center(0.5, 0.5);
    Point<dim - 1> face_midpoint(0.5);
    Quadrature<dim> quadrature_formula(cell_center);
    Quadrature<dim - 1> quadrature_formula_face(face_midpoint);
//    QGauss<dim> quadrature_formula(2);
//    QGauss<dim - 1> quadrature_formula_face(2);

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

    FullMatrix<double> cell_matrix_M(dofs_per_cell_h, dofs_per_cell_h);
    Vector<double> cell_vector_G(dofs_per_cell_h);
    Vector<double> cell_dw2pd_coeffs(dofs_per_cell_h);

    Cos_pi_x_Cos_pi_y<dim> cos_pi_x_cos_pi_y;
    Sin_pi_x_Sin_pi_y<dim> sin_pi_x_sin_pi_y;
    X_2_Y_2<dim> x_2_y_2;

    std::vector<Point<3>> errors;


    /***************************************************************************
     ***************************************************************************/

    // Setup cell iterators
    typename DoFHandler<dim>::active_cell_iterator
        cell   = dof_handler.begin_active(),
        cell_f = dof_handler_f.begin_active(),
        cell_h = dof_handler_h.begin_active(),
        endc   = dof_handler_f.end();

    for (; cell_f != endc; ++cell, ++cell_f, ++cell_h) {

        fe_values_f.reinit(cell_f);
        fe_values_h.reinit(cell_h);

        // Ensure correct interpolation
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

        // Record error in the interpolation
//        std::vector<double> function_vals(n_quad_points);
//        fe_values_f.get_function_values(solution, interior_dof_indices, function_vals);
//
//        const auto points = fe_values_f.get_quadrature_points();
//        for (unsigned int q = 0; q < n_quad_points; ++q) {
//            Point<3> error(points[q](0),
//                           points[q](1),
//                           std::abs(function_vals[q] - sin_pi_x_sin_pi_y.value(points[q], 0)));
//            std::cout << function_vals[q] << " : " << sin_pi_x_sin_pi_y.value(points[q], 0) << " : " << error << std::endl;
//            errors.push_back(error);
//        }

        // TODO: Build mass matrix
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
        cell_matrix_M.gauss_jordan();

        FullMatrix<double> cell_hessian(dim, dim);

        for (unsigned int d1 = 0; d1 < dim; ++d1) {
            for (unsigned int d2 = 0; d2 < dim; ++d2) {

                cell_vector_G = 0;
                cell_dw2pd_coeffs = 0;

                // TODO: Incorporate interior part of G (later)

                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f) {

                    fe_values_f_face.reinit(cell_f, cell_f->face(f));
                    fe_values_h_face.reinit(cell_h, cell_h->face(f));

                    // TODO: Incorporate face part of G (later)


                    // TODO: Incorporate face gradient part of G

                    std::vector<Tensor<1, dim>> function_gradients(n_quad_points_face);
                    fe_values_f_face.get_function_gradients(solution, interior_dof_indices, function_gradients);

                    // Take average of gradient at the neighboring cell
//                    if (cell_f->at_boundary(f) == false) {
//                        const auto neighbor = cell_f->neighbor(f);
//                        fe_values_f_face.reinit(neighbor, cell_f->face(f));
//                        std::vector<Tensor<1, dim>> neighbor_function_gradients(n_quad_points_face);
//                        fe_values_f_face.get_function_gradients(function_coeffs, neighbor_function_gradients);
//
//                        for (unsigned q = 0; q < n_quad_points_face; ++q) {
//                            function_gradients[q] = 0.5 * (function_gradients[q] + neighbor_function_gradients[q]);
//                        }
//                    }

                    // Record error in function gradient
//                    const auto points = fe_values_f_face.get_quadrature_points();
//                    for (unsigned int q = 0; q < n_quad_points_face; ++q) {
//                        Point<3> error(points[q](0), points[q](1),
//                                       std::abs(function_gradients[q][0]- sin_pi_x_sin_pi_y.gradient(points[q], 0)[0]));
//                        errors.push_back(error);
//                        Point<3> error1(points[q](0), points[q](1),
//                                       std::abs(function_gradients[q][1]- sin_pi_x_sin_pi_y.gradient(points[q], 0)[1]));
//                        errors.push_back(error1);
//                    }

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

                // TODO: Compute cell_dw2pd

                cell_matrix_M.vmult(cell_dw2pd_coeffs, cell_vector_G);
                std::vector<double> cell_dw2pds(n_quad_points);
                std::vector<types::global_dof_index> dof_indices_h = {0};
                fe_values_h.get_function_values(cell_dw2pd_coeffs, dof_indices_h, cell_dw2pds);

                cell_hessian(d1, d2) = cell_dw2pds[0];

                // Record error in function hessian
//                const auto points = fe_values_h.get_quadrature_points();
//                for (unsigned int q = 0; q < n_quad_points; ++q) {
//                    Point<3> error(points[q](0), points[q](1),
//                                   std::abs(cell_dw2pds[q] - cos_pi_x_cos_pi_y.hessian(points[q], 0)[d1][d2]));
//                    errors.push_back(error);
//                }
            }
        }

        hessian_det[cell->active_cell_index()] = cell_hessian.determinant();

        const auto point = fe_values_h.get_quadrature_points()[0];
        Point<3> error(point(0), point(1), std::abs(cell_hessian.determinant() - cos_pi_x_cos_pi_y.hessian_det(point)));
        errors.push_back(error);
    }

    // Output error data for visualaization
    std::remove("./error_z.txt");
    std::ofstream output_file_z("./error_z.txt");
    std::ostream_iterator<Point<3>> output_iterator(output_file_z, "\n");
    std::copy(errors.begin(), errors.end(), output_iterator);
}


template <int dim>
void WGOptimalTransport<dim>::compute_pressure_error()
{
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    const ComponentSelectFunction<dim> select_interior_pressure(0, 2);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Cos_pi_x_Cos_pi_y<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm,
                                      &select_interior_pressure);
    const double L2_error = difference_per_cell.l2_norm();
    std::cout << "L2_error_pressure " << L2_error << std::endl;
}

// @sect4{WGOptimalTransport::run}

// This is the final function of the main class. It calls the other functions
// of our class.
template <int dim>
void WGOptimalTransport<dim>::run()
{
    make_grid(4);
    setup_system();
    assemble_system_matrix();
    assemble_system_rhs();
    // Example functions
//    Cos_pi_x_Cos_pi_y<dim> cos_pi_x_cos_pi_y;
//    Sin_pi_x_Sin_pi_y<dim> sin_pi_x_sin_pi_y;
//    X_2_Y_2<dim> x_2_y_2;
//    VectorTools::interpolate(dof_handler, cos_pi_x_cos_pi_y, solution);
    solve();
    compute_hessian();
    compute_pressure_error();
//    for (unsigned int i = 0; i < 5; ++i) {
//        compute_hessian();
//        assemble_system_rhs();
//        solve();
//        compute_pressure_error();
//    }
//    std::ofstream file_out("sin_sin_5_refs.txt");
//    solution.block_write(file_out);
//    std::ifstream file_in("cos_cos_3_refs.txt");
//    solution.block_read(file_in);
//    output_results();
}