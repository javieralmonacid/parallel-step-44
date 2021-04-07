/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2021 by the deal.II authors and
 *                              & Javier A. Almonacid
 *
 * This file is *NOT* part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Javier A. Almonacid, Simon Fraser University, 2021
 *         
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
//#include <deal.II/base/work_stream.h> // we shouldn't need this
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

// This header gives us the functionality to store data at quadrature points
#include <deal.II/base/quadrature_point_data.h>

//#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h> // not in step-44
//#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/affine_constraints.h> // NEW
#include <deal.II/lac/constraint_matrix.h> // FROM STEP-44
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/petsc_parallel_block_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_block_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>

#include "parameter-namespace.h"

namespace Step44{
    using namespace dealii;

    class Time
    {
    public:
        Time (const double time_end, const double delta_t)
            :
            timestep(0),
            time_current(0.0),
            time_end(time_end),
            delta_t(delta_t)
            {}

        virtual ~Time()
        {}

        double current() const
        {
        return time_current;
        }
        double end() const
        {
        return time_end;
        }
        double get_delta_t() const
        {
        return delta_t;
        }
        unsigned int get_timestep() const
        {
        return timestep;
        }
        void increment()
        {
        time_current += delta_t;
        ++timestep;
        }

    private:
        unsigned int timestep;
        double       time_current;
        const double time_end;
        const double delta_t;
    };

    // @sect3{Compressible neo-Hookean material within a three-field formulation} 
    template <int dim>
    class Material_Compressible_Neo_Hook_Three_Field
    {
    public:
        Material_Compressible_Neo_Hook_Three_Field(const double mu,
                                                const double nu)
        :
        kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
        c_1(mu / 2.0),
        det_F(1.0),
        p_tilde(0.0),
        J_tilde(1.0),
        b_bar(Physics::Elasticity::StandardTensors<dim>::I)
        {
        Assert(kappa > 0, ExcInternalError());
        }

        ~Material_Compressible_Neo_Hook_Three_Field()
        {}

        // We update the material model with various deformation dependent data
        // based on $F$ and the pressure $\widetilde{p}$ and dilatation
        // $\widetilde{J}$, and at the end of the function include a physical
        // check for internal consistency:
        void update_material_data(const Tensor<2, dim> &F,
                                const double p_tilde_in,
                                const double J_tilde_in)
        {
        det_F = determinant(F);
        const Tensor<2, dim> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
        b_bar = Physics::Elasticity::Kinematics::b(F_bar);
        p_tilde = p_tilde_in;
        J_tilde = J_tilde_in;

        Assert(det_F > 0, ExcInternalError());
        }

        // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
        // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
        SymmetricTensor<2, dim> get_tau()
        {
        return get_tau_iso() + get_tau_vol();
        }

        // The fourth-order elasticity tensor in the spatial setting
        // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
        // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
        // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
        // \mathbf{C} \partial \mathbf{C}}$
        SymmetricTensor<4, dim> get_Jc() const
        {
        return get_Jc_vol() + get_Jc_iso();
        }

        // Derivative of the volumetric free energy with respect to
        // $\widetilde{J}$ return $\frac{\partial
        // \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
        double get_dPsi_vol_dJ() const
        {
        return (kappa / 2.0) * (J_tilde - 1.0 / J_tilde);
        }

        // Second derivative of the volumetric free energy wrt $\widetilde{J}$. We
        // need the following computation explicitly in the tangent so we make it
        // public.  We calculate $\frac{\partial^2
        // \Psi_{\textrm{vol}}(\widetilde{J})}{\partial \widetilde{J} \partial
        // \widetilde{J}}$
        double get_d2Psi_vol_dJ2() const
        {
        return ( (kappa / 2.0) * (1.0 + 1.0 / (J_tilde * J_tilde)));
        }

        // The next few functions return various data that we choose to store with
        // the material:
        double get_det_F() const
        {
        return det_F;
        }

        double get_p_tilde() const
        {
        return p_tilde;
        }

        double get_J_tilde() const
        {
        return J_tilde;
        }

    protected:
        // Define constitutive model parameters $\kappa$ (bulk modulus) and the
        // neo-Hookean model parameter $c_1$:
        const double kappa;
        const double c_1;

        // Model specific data that is convenient to store with the material:
        double det_F;
        double p_tilde;
        double J_tilde;
        SymmetricTensor<2, dim> b_bar;

        // The following functions are used internally in determining the result
        // of some of the public functions above. The first one determines the
        // volumetric Kirchhoff stress $\boldsymbol{\tau}_{\textrm{vol}}$:
        SymmetricTensor<2, dim> get_tau_vol() const
        {
        return p_tilde * det_F * Physics::Elasticity::StandardTensors<dim>::I;
        }

        // Next, determine the isochoric Kirchhoff stress
        // $\boldsymbol{\tau}_{\textrm{iso}} =
        // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
        SymmetricTensor<2, dim> get_tau_iso() const
        {
        return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar();
        }

        // Then, determine the fictitious Kirchhoff stress
        // $\overline{\boldsymbol{\tau}}$:
        SymmetricTensor<2, dim> get_tau_bar() const
        {
        return 2.0 * c_1 * b_bar;
        }

        // Calculate the volumetric part of the tangent $J
        // \mathfrak{c}_\textrm{vol}$:
        SymmetricTensor<4, dim> get_Jc_vol() const
        {

        return p_tilde * det_F
                * ( Physics::Elasticity::StandardTensors<dim>::IxI
                    - (2.0 * Physics::Elasticity::StandardTensors<dim>::S) );
        }

        // Calculate the isochoric part of the tangent $J
        // \mathfrak{c}_\textrm{iso}$:
        SymmetricTensor<4, dim> get_Jc_iso() const
        {
        const SymmetricTensor<2, dim> tau_bar = get_tau_bar();
        const SymmetricTensor<2, dim> tau_iso = get_tau_iso();
        const SymmetricTensor<4, dim> tau_iso_x_I
            = outer_product(tau_iso,
                            Physics::Elasticity::StandardTensors<dim>::I);
        const SymmetricTensor<4, dim> I_x_tau_iso
            = outer_product(Physics::Elasticity::StandardTensors<dim>::I,
                            tau_iso);
        const SymmetricTensor<4, dim> c_bar = get_c_bar();

        return (2.0 / dim) * trace(tau_bar)
                * Physics::Elasticity::StandardTensors<dim>::dev_P
                - (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso)
                + Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar
                * Physics::Elasticity::StandardTensors<dim>::dev_P;
        }

        // Calculate the fictitious elasticity tensor $\overline{\mathfrak{c}}$.
        // For the material model chosen this is simply zero:
        SymmetricTensor<4, dim> get_c_bar() const
        {
        return SymmetricTensor<4, dim>();
        }
    }; // This was not modified from step-44

    // @sect3{Quadrature point history}

// As seen in step-18, the <code> PointHistory </code> class offers a method
// for storing data at the quadrature points.  Here each quadrature point
// holds a pointer to a material description.  Thus, different material models
// can be used in different regions of the domain.  Among other data, we
// choose to store the Kirchhoff stress $\boldsymbol{\tau}$ and the tangent
// $J\mathfrak{c}$ for the quadrature points.
    template <int dim>
    class PointHistory
    {
    public:
        PointHistory()
        :
        F_inv(Physics::Elasticity::StandardTensors<dim>::I),
        tau(SymmetricTensor<2, dim>()),
        d2Psi_vol_dJ2(0.0),
        dPsi_vol_dJ(0.0),
        Jc(SymmetricTensor<4, dim>())
        {}

        virtual ~PointHistory()
        {}

        // The first function is used to create a material object and to
        // initialize all tensors correctly: The second one updates the stored
        // values and stresses based on the current deformation measure
        // $\textrm{Grad}\mathbf{u}_{\textrm{n}}$, pressure $\widetilde{p}$ and
        // dilation $\widetilde{J}$ field values.
        void setup_lqp (const Parameters::AllParameters &parameters)
        {
        material.reset(new Material_Compressible_Neo_Hook_Three_Field<dim>(parameters.mu,
                        parameters.nu));
        update_values(Tensor<2, dim>(), 0.0, 1.0);
        }

        // To this end, we calculate the deformation gradient $\mathbf{F}$ from
        // the displacement gradient $\textrm{Grad}\ \mathbf{u}$, i.e.
        // $\mathbf{F}(\mathbf{u}) = \mathbf{I} + \textrm{Grad}\ \mathbf{u}$ and
        // then let the material model associated with this quadrature point
        // update itself. When computing the deformation gradient, we have to take
        // care with which data types we compare the sum $\mathbf{I} +
        // \textrm{Grad}\ \mathbf{u}$: Since $I$ has data type SymmetricTensor,
        // just writing <code>I + Grad_u_n</code> would convert the second
        // argument to a symmetric tensor, perform the sum, and then cast the
        // result to a Tensor (i.e., the type of a possibly nonsymmetric
        // tensor). However, since <code>Grad_u_n</code> is nonsymmetric in
        // general, the conversion to SymmetricTensor will fail. We can avoid this
        // back and forth by converting $I$ to Tensor first, and then performing
        // the addition as between nonsymmetric tensors:
        void update_values (const Tensor<2, dim> &Grad_u_n,
                            const double p_tilde,
                            const double J_tilde)
        {
        const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u_n);
        material->update_material_data(F, p_tilde, J_tilde);

        // The material has been updated so we now calculate the Kirchhoff
        // stress $\mathbf{\tau}$, the tangent $J\mathfrak{c}$ and the first and
        // second derivatives of the volumetric free energy.
        //
        // We also store the inverse of the deformation gradient since we
        // frequently use it:
        F_inv = invert(F);
        tau = material->get_tau();
        Jc = material->get_Jc();
        dPsi_vol_dJ = material->get_dPsi_vol_dJ();
        d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();

        }

        // We offer an interface to retrieve certain data.  Here are the kinematic
        // variables:
        double get_J_tilde() const
        {
        return material->get_J_tilde();
        }

        double get_det_F() const
        {
        return material->get_det_F();
        }

        const Tensor<2, dim> &get_F_inv() const
        {
        return F_inv;
        }

        // ...and the kinetic variables.  These are used in the material and
        // global tangent matrix and residual assembly operations:
        double get_p_tilde() const
        {
        return material->get_p_tilde();
        }

        const SymmetricTensor<2, dim> &get_tau() const
        {
        return tau;
        }

        double get_dPsi_vol_dJ() const
        {
        return dPsi_vol_dJ;
        }

        double get_d2Psi_vol_dJ2() const
        {
        return d2Psi_vol_dJ2;
        }

        // And finally the tangent:
        const SymmetricTensor<4, dim> &get_Jc() const
        {
        return Jc;
        }

        // In terms of member functions, this class stores for the quadrature
        // point it represents a copy of a material type in case different
        // materials are used in different regions of the domain, as well as the
        // inverse of the deformation gradient...
    private:
        std_cxx11::shared_ptr< Material_Compressible_Neo_Hook_Three_Field<dim> > material;

        Tensor<2, dim> F_inv;

        // ... and stress-type variables along with the tangent $J\mathfrak{c}$:
        SymmetricTensor<2, dim> tau;
        double                  d2Psi_vol_dJ2;
        double                  dPsi_vol_dJ;

        SymmetricTensor<4, dim> Jc;
    };

// @sect3{Quasi-static quasi-incompressible finite-strain solid} ==========================================

// The Solid class is the central class in that it represents the problem at
// hand. It follows the usual scheme in that all it really has is a
// constructor, destructor and a <code>run()</code> function that dispatches
// all the work to private functions of this class:

    template <int dim>
    class Solid
    {
    public:
        Solid(const std::string &input_file);
        virtual ~Solid();
        void run();
    
    private:
        // In the private section of this class, we first forward declare a number
        // of objects that are used in parallelizing work using the WorkStream
        // object (see the @ref threads module for more information on this).
        //
        // We declare such structures for the computation of tangent (stiffness)
        // matrix, right hand side, static condensation, and for updating
        // quadrature points:
        // ==================> WE WILL RECYCLE THIS
        struct PerTaskData_K;
        struct ScratchData_K;

        struct PerTaskData_RHS;
        struct ScratchData_RHS;

        struct PerTaskData_SC;
        struct ScratchData_SC;

        struct PerTaskData_UQPH;
        struct ScratchData_UQPH;

        void make_grid();
        void setup_system();
        void determine_component_extractors();

        void assemble_system_tangent();
        void assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_K &scratch,
                                              PerTaskData_K &data) const;
        void copy_local_to_global_K(const PerTaskData_K &data);

        void assemble_system_rhs();
        void assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          ScratchData_RHS &scratch,
                                          PerTaskData_RHS &data);
        void copy_local_to_global_rhs(const PerTaskData_RHS &data);

        void assemble_sc();
        void assemble_sc_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  ScratchData_SC &scratch,
                                  PerTaskData_SC &data);
        void copy_local_to_global_sc(const PerTaskData_SC &data);

        // Apply Dirichlet BC on the displacement
        void make_constraints(const int &it_nr);

        // Create and update the quadrature points. Here, no data needs to be
        // copied into a global object, so the copy_local_to_global function is
        // empty:
        void setup_qph();

        void update_qph_incremental(const PETScWrappers::MPI::BlockVector &solution_delta);
        void update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             ScratchData_UQPH &scratch,
                                             PerTaskData_UQPH &data);
        void copy_local_to_global_UQPH(const PerTaskData_UQPH &/*data*/)
        {}

        // Solve for the displacement using a Newton-Raphson method. We break this
        // function into the nonlinear loop and the function that solves the
        // linearized Newton-Raphson step:
        void solve_nonlinear_timestep(PETScWrappers::MPI::BlockVector &solution_delta);
        std::pair<unsigned int, double> 
        solve_linear_system(PETScWrappers::MPI::BlockVector &newton_update);
        PETScWrappers::MPI::BlockVector 
        get_total_solution(const PETScWrappers::MPI::BlockVector &solution_delta);

        void output_results() const;

        // NEW: MPI related variables ===========================
        MPI_Comm mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
        mutable ConditionalOStream pcout;

        // Finally, some member variables that describe the current state: A
        // collection of the parameters used to describe the problem setup...
        Parameters::AllParameters parameters;

        // ...the volume of the reference configuration...
        double vol_reference;

        // ...and description of the geometry on which the problem is solved:
        parallel::shared::Triangulation<dim> triangulation;

        // Also, keep track of the current time and the time spent evaluating
        // certain functions
        Time                time;
        mutable TimerOutput timer;

        // A storage object for quadrature point information. As opposed to
        // step-18, deal.II's native quadrature point data manager is employed
        // here.
        CellDataStorage<typename Triangulation<dim>::cell_iterator,
                        PointHistory<dim> > quadrature_point_history; // PROBABLY THIS CELL ITERATOR IS ENOUGH

        // A description of the finite-element system including the displacement
        // polynomial degree, the degree-of-freedom handler, number of DoFs per
        // cell and the extractor objects used to retrieve information from the
        // solution vectors:
        const unsigned int               degree;
        const FESystem<dim>              fe;
        DoFHandler<dim>                  dof_handler_ref;
        const unsigned int               dofs_per_cell;
        const FEValuesExtractors::Vector u_fe;
        const FEValuesExtractors::Scalar p_fe;
        const FEValuesExtractors::Scalar J_fe;

        // Description of how the block-system is arranged. There are 3 blocks,
        // the first contains a vector DOF $\mathbf{u}$ while the other two
        // describe scalar DOFs, $\widetilde{p}$ and $\widetilde{J}$.
        static const unsigned int n_blocks = 3;
        static const unsigned int n_components = dim + 2;
        static const unsigned int first_u_component = 0;
        static const unsigned int p_component = dim;
        static const unsigned int J_component = dim + 1;

        enum
        {
        u_dof = 0,
        p_dof = 1,
        J_dof = 2
        };

        std::vector<types::global_dof_index> dofs_per_block;
        std::vector<types::global_dof_index> element_indices_u;
        std::vector<types::global_dof_index> element_indices_p;
        std::vector<types::global_dof_index> element_indices_J;

        // NEW: MPI RELATED ======================================
        std::vector<unsigned int> block_component;
        std::vector<IndexSet> all_locally_owned_dofs;
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> locally_owned_partitioning;
        std::vector<IndexSet> locally_relevant_partitioning;

        // Rules for Gauss-quadrature on both the cell and faces. The number of
        // quadrature points on both cells and faces is recorded.
        const QGauss<dim>     qf_cell;
        const QGauss<dim - 1> qf_face;
        const unsigned int    n_q_points;
        const unsigned int    n_q_points_f;

        // CAUTION ===================================================================================================================
        // Objects that store the converged solution and right-hand side vectors,
        // as well as the tangent matrix. There is a ConstraintMatrix object used
        // to keep track of constraints.  We make use of a sparsity pattern
        // designed for a block system.
        ConstraintMatrix                      constraints;
        BlockSparsityPattern                  sparsity_pattern;
        PETScWrappers::MPI::BlockSparseMatrix tangent_matrix;
        PETScWrappers::MPI::BlockVector       system_rhs;
        PETScWrappers::MPI::BlockVector       solution_n;

        // Then define a number of variables to store norms and update norms and
        // normalisation factors.
        struct Errors
        {
            Errors()
                :
                norm(1.0), u(1.0), p(1.0), J(1.0)
            {}

            void reset()
            {
                norm = 1.0;
                u = 1.0;
                p = 1.0;
                J = 1.0;
            }
            void normalise(const Errors &rhs)
            {
                if (rhs.norm != 0.0)
                norm /= rhs.norm;
                if (rhs.u != 0.0)
                u /= rhs.u;
                if (rhs.p != 0.0)
                p /= rhs.p;
                if (rhs.J != 0.0)
                J /= rhs.J;
            }

            double norm, u, p, J;
        };

        Errors error_residual, error_residual_0, error_residual_norm, error_update,
            error_update_0, error_update_norm;

        // Methods to calculate error measures
        void get_error_residual(Errors &error_residual);
        void get_error_update(const PETScWrappers::MPI::BlockVector &newton_update, Errors &error_update);
        std::pair<double, double> get_error_dilation() const;

        // Compute the volume in spatial configuration
        double compute_vol_current() const;

        // Print information to screen in a pleasing way
        static void print_conv_header();
        void print_conv_footer();        
    };

// @sect3{Implementation of the <code>Solid</code> class}

// @sect4{Public interface}

// We initialise the Solid class using data extracted from the parameter file.
    template <int dim>
    Solid<dim>::Solid(const std::string &input_file):
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, this_mpi_process == 0),
        parameters(input_file),
        vol_reference(0.),
        //triangulation(Triangulation<dim>::maximum_smoothing),
        triangulation(mpi_communicator, typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::maximum_smoothing)),
        time(parameters.end_time, parameters.delta_t),
        timer(mpi_communicator,
            pcout,
            TimerOutput::summary,
            TimerOutput::wall_times),
        degree(parameters.poly_degree),
        // The Finite Element System is composed of dim continuous displacement
        // DOFs, and discontinuous pressure and dilatation DOFs. In an attempt to
        // satisfy the Babuska-Brezzi or LBB stability conditions (see Hughes
        // (2000)), we setup a $Q_n \times DGPM_{n-1} \times DGPM_{n-1}$
        // system. $Q_2 \times DGPM_1 \times DGPM_1$ elements satisfy this
        // condition, while $Q_1 \times DGPM_0 \times DGPM_0$ elements do
        // not. However, it has been shown that the latter demonstrate good
        // convergence characteristics nonetheless.
        fe(FE_Q<dim>(parameters.poly_degree), dim, // displacement
        FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1, // pressure
        FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1), // dilatation
        dof_handler_ref(triangulation),
        dofs_per_cell (fe.dofs_per_cell),
        u_fe(first_u_component),
        p_fe(p_component),
        J_fe(J_component),
        dofs_per_block(n_blocks),
        qf_cell(parameters.quad_order),
        qf_face(parameters.quad_order),
        n_q_points (qf_cell.size()),
        n_q_points_f (qf_face.size())
    {
        Assert(dim==2 || dim==3, ExcMessage("This problem only works in 2 or 3 space dimensions."));
        //determine_component_extractors();
    }

    // The class destructor simply clears the data held by the DOFHandler
    template <int dim>
    Solid<dim>::~Solid()
    {
        dof_handler_ref.clear();
    }

    template <int dim>
    void Solid<dim>::run()
    {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
            std::cout << "TESTING TESTING TESTING" << std::endl;
        }
    }

// @sect3{Private interface}

// @sect4{Threading-building-blocks structures}

// The first group of private member functions is related to parallization.
// We use the Threading Building Blocks library (TBB) to perform as many
// computationally intensive distributed tasks as possible. In particular, we
// assemble the tangent matrix and right hand side vector, the static
// condensation contributions, and update data stored at the quadrature points
// using TBB. Our main tool for this is the WorkStream class (see the @ref
// threads module for more information).

// Firstly we deal with the tangent matrix assembly structures.  The
// PerTaskData object stores local contributions.
    template <int dim>
    struct Solid<dim>::PerTaskData_K
    {
        FullMatrix<double>        cell_matrix;
        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_K(const unsigned int dofs_per_cell):
            cell_matrix(dofs_per_cell, dofs_per_cell),
            local_dof_indices(dofs_per_cell)
        {}

        void reset()
        {
        cell_matrix = 0.0;
        }
    };

// On the other hand, the ScratchData object stores the larger objects such as
// the shape-function values array (<code>Nx</code>) and a shape function
// gradient and symmetric gradient vector which we will use during the
// assembly.

    template <int dim>
    struct Solid<dim>::ScratchData_K
    {
        FEValues<dim> fe_values_ref;

        // Shape function values and gradients
        std::vector<std::vector<double> >                   Nx;
        std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;
        std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;

        ScratchData_K(const FiniteElement<dim> &fe_cell,
                      const QGauss<dim> &qf_cell,
                      const UpdateFlags uf_cell):
            fe_values_ref(fe_cell, qf_cell, uf_cell),
            Nx(qf_cell.size(), 
               std::vector<double>(fe_cell.dofs_per_cell)),
            grad_Nx(qf_cell.size(), 
                    std::vector<Tensor<2, dim> >(fe_cell.dofs_per_cell)),
            symm_grad_Nx(qf_cell.size(),
                        std::vector<SymmetricTensor<2, dim> >
                        (fe_cell.dofs_per_cell))
        {}

        ScratchData_K(const ScratchData_K &rhs):
            fe_values_ref(rhs.fe_values_ref.get_fe(),
                          rhs.fe_values_ref.get_quadrature(),
                          rhs.fe_values_ref.get_update_flags()),
            Nx(rhs.Nx),
            grad_Nx(rhs.grad_Nx),
            symm_grad_Nx(rhs.symm_grad_Nx)
        {}

        void reset()
        {
            const unsigned int n_q_points = Nx.size();
            const unsigned int n_dofs_per_cell = Nx[0].size();

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
                Assert( Nx[q_point].size() == n_dofs_per_cell, 
                        ExcInternalError());
                Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                        ExcInternalError());
                Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                        ExcInternalError());
                
                for (unsigned int k = 0; k < n_dofs_per_cell; ++k){
                    Nx[q_point][k] = 0.0;
                    grad_Nx[q_point][k] = 0.0;
                    symm_grad_Nx[q_point][k] = 0.0;
                }
            }
        }
    };

// Next, the same approach is used for the right-hand side assembly.  The
// PerTaskData object again stores local contributions and the ScratchData
// object the shape function object and precomputed values vector
    template <int dim>
    struct Solid<dim>::PerTaskData_RHS
    {
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_RHS(const unsigned int dofs_per_cell):
            cell_rhs(dofs_per_cell),
            local_dof_indices(dofs_per_cell)
        {}

        void reset()
        {
            cell_rhs = 0.0;
        }
    };

    template <int dim>
    struct Solid<dim>::ScratchData_RHS
    {
        FEValues<dim> fe_values_ref;
        FEFaceValues<dim> fe_face_values_ref;

        std::vector<std::vector<double> >                   Nx;
        std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;

        ScratchData_RHS(const FiniteElement<dim> &fe_cell,
                        const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                        const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face):
            fe_values_ref(fe_cell, qf_cell, uf_cell),
            fe_face_values_ref(fe_cell, qf_face, uf_face),
            Nx(qf_cell.size(),
               std::vector<double>(fe_cell.dofs_per_cell)),
            symm_grad_Nx(qf_cell.size(),
                         std::vector<SymmetricTensor<2, dim> >(fe_cell.dofs_per_cell))
        {}

        ScratchData_RHS(const ScratchData_RHS &rhs):
            fe_values_ref(rhs.fe_values_ref.get_fe(),
                          rhs.fe_values_ref.get_quadrature(),
                          rhs.fe_values_ref.get_update_flags()),
            fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                               rhs.fe_face_values_ref.get_quadrature(),
                               rhs.fe_face_values_ref.get_update_flags()),
            Nx(rhs.Nx),
            symm_grad_Nx(rhs.symm_grad_Nx)
        {}

        void reset()
        {
            const unsigned int n_q_points = Nx.size();
            const unsigned int n_dofs_per_cell = Nx[0].size();

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
                Assert( Nx[q_point].size() == n_dofs_per_cell, 
                        ExcInternalError());
                Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                        ExcInternalError());
                for (unsigned int k = 0; k < n_dofs_per_cell; ++k){
                    Nx[q_point][k] = 0.0;
                    symm_grad_Nx[q_point][k] = 0.0;
                }
            }
        }
            
    };
    

}


int main(int argc, char *argv[])
{
    using namespace dealii;
    using namespace Step44;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

    try
    {
        const unsigned int dim = 3;
        Solid<dim> solid("parameters.prm");
        solid.run();
    }
    catch (std::exception &exc)
    {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
            std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
            std::cerr << "Exception on processing: " << std::endl
                    << exc.what() << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;

            return 1;
        }
    }
    catch (...)
    {
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
            std::cerr << std::endl << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
            std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                        << std::endl
                        << "----------------------------------------------------"
                        << std::endl;
            return 1;
        }
    }

    return 0;
}