/* ---------------------------------------------------------------------
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
//#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

// This header gives us the functionality to store data at quadrature points
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h> 
//#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h> // FROM STEP-44
#include <deal.II/lac/sparsity_tools.h> // FROM STEP-55
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_selector.h>

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

namespace Step44
{
    using namespace dealii;
    
    class Time
    {
    public:
        Time (const double time_end, const double delta_t):
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

    template <int dim>
    class Material_Compressible_Neo_Hook_Three_Field
    {
    public:
        Material_Compressible_Neo_Hook_Three_Field(const double mu, const double nu):
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
            return p_tilde * det_F * ( Physics::Elasticity::StandardTensors<dim>::IxI
                                      - (2.0 * Physics::Elasticity::StandardTensors<dim>::S) );
        }

        // Calculate the isochoric part of the tangent $J
        // \mathfrak{c}_\textrm{iso}$:
        SymmetricTensor<4, dim> get_Jc_iso() const
        {
            const SymmetricTensor<2, dim> tau_bar = get_tau_bar();
            const SymmetricTensor<2, dim> tau_iso = get_tau_iso();
            const SymmetricTensor<4, dim> tau_iso_x_I
                = outer_product(tau_iso, Physics::Elasticity::StandardTensors<dim>::I);
            const SymmetricTensor<4, dim> I_x_tau_iso
                = outer_product(Physics::Elasticity::StandardTensors<dim>::I, tau_iso);
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
        PointHistory():
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
        // of objects that are used in parallelizing work.
        //
        // We declare such structures for the computation of tangent (stiffness)
        // matrix, right hand side, and for updating quadrature points:
        struct PerTaskData_ASM;
        struct ScratchData_ASM;
        struct ScratchData_UQPH;

        void make_grid();
        void system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta);
        void set_initial_dilation(TrilinosWrappers::MPI::BlockVector &solution_n_relevant);
        void determine_component_extractors();

        void assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_delta);
        void assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                      ScratchData_ASM &scratch,
                                      PerTaskData_ASM &data) const;
        void copy_local_to_global_system(const PerTaskData_ASM &data);

        // Apply Dirichlet BC on the displacement
        void make_constraints(const int &it_nr);

        // Create and update the quadrature points. Here, no data needs to be
        // copied into a global object, so the copy_local_to_global function is
        // empty:
        void setup_qph();
        void update_qph(TrilinosWrappers::MPI::BlockVector &solution_delta);
        void update_qph_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 ScratchData_UQPH &scratch);

        void solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta);
        std::pair<unsigned int, double> solve_linear_system(TrilinosWrappers::MPI::BlockVector &newton_update);
        TrilinosWrappers::MPI::BlockVector get_solution_total(const TrilinosWrappers::MPI::BlockVector &solution_delta) const;
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
        // ==== parallel::shared vs parallel::distributed
        parallel::distributed::Triangulation<dim> triangulation;

        // Also, keep track of the current time and the time spent evaluating
        // certain functions
        Time                time;
        mutable TimerOutput  timer;

        // A storage object for quadrature point information. As opposed to
        // step-18, deal.II's native quadrature point data manager is employed
        // here.
        CellDataStorage<typename Triangulation<dim>::cell_iterator,
                        PointHistory<dim> > quadrature_point_history;
        
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
            u_block = 0,
            p_block = 1,
            J_block = 2
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

        // Objects that store the converged solution and right-hand side vectors,
        // as well as the tangent matrix. There is a ConstraintMatrix object used
        // to keep track of constraints.
        ConstraintMatrix                        constraints;
        TrilinosWrappers::BlockSparseMatrix     tangent_matrix;
        TrilinosWrappers::MPI::BlockVector      system_rhs;
        TrilinosWrappers::MPI::BlockVector      solution_n_relevant;

        // Then define a number of variables to store norms and update norms and
        // normalisation factors.
        struct Errors
        {
            Errors():
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
        void get_error_update(const TrilinosWrappers::MPI::BlockVector &solution_total, 
                              Errors &error_update);
        std::pair<double, double> get_error_dilation() const;

        // Compute the volume in spatial configuration
        double compute_vol_current() const;

        // Print information to screen in a pleasing way
        void print_conv_header();
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
        triangulation(mpi_communicator,
                        typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening)),
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
        fe (FE_Q<dim>(parameters.poly_degree), dim,              // displacement
            FE_DGPMonomial<dim>(parameters.poly_degree - 1), 1,  // pressure
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
        determine_component_extractors();
    }

    // The class destructor simply clears the data held by the DOFHandler
    template <int dim>
    Solid<dim>::~Solid()
    {
        dof_handler_ref.clear();
    }

// @sect4{Solid::determine_component_extractors}
// Next we compute some information from the FE system that describes which local
// element DOFs are attached to which block component.  This is used later to
// extract sub-blocks from the global matrix.
//
// In essence, all we need is for the FESystem object to indicate to which
// block component a DOF on the reference cell is attached to.  Currently, the
// interpolation fields are setup such that 0 indicates a displacement DOF, 1
// a pressure DOF and 2 a dilatation DOF.
    template <int dim>
    void Solid<dim>::determine_component_extractors()
    {
        element_indices_u.clear();
        element_indices_p.clear();
        element_indices_J.clear();

        for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
        {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
                element_indices_u.push_back(k);
            else if (k_group == p_block)
                element_indices_p.push_back(k);
            else if (k_group == J_block)
                element_indices_J.push_back(k);
            else{
                Assert(k_group <= J_block, ExcInternalError());
            }
        }
    }

// In solving the quasi-static problem, the time becomes a loading parameter,
// i.e. we increase the loading linearly with time, making the two concepts
// interchangeable. We choose to increment time linearly using a constant time
// step size.
//
// We start the function with preprocessing, setting the initial dilatation
// values, and then output the initial grid before starting the simulation
//  proper with the first time (and loading)
// increment.
    template <int dim>
    void Solid<dim>::run()
    {
    	pcout << "********* PARALLEL STEP 44 *********" << std::endl;
	    pcout << "Running with " << Utilities::MPI::n_mpi_processes(mpi_communicator) 
              << " processes" << std::endl;
	    pcout << std::endl;
        // We first declare the incremental solution update $\varDelta
        // \mathbf{\Xi}:= \{\varDelta \mathbf{u},\varDelta \widetilde{p},
        // \varDelta \widetilde{J} \}$.
        TrilinosWrappers::MPI::BlockVector solution_delta;
        
        make_grid();
	    pcout << "Total number of active cells: " 
              << triangulation.n_global_active_cells() << std::endl;

        // The following step will initialize solution_delta and solution_n_relevant
        system_setup(solution_delta); 

        // Care must be taken (or at least some thought given) when imposing the
        // constraint $\widetilde{J}=1$ on the initial solution field. The constraint
        // corresponds to the determinant of the deformation gradient in the undeformed
        // configuration, which is the identity tensor.
        // We use FE_DGPMonomial bases to interpolate the dilatation field, thus we can't
        // simply set the corresponding dof to unity as they correspond to the
        // monomial coefficients. The VectorTools::project functions does this work
        // automatically, but this function does not work in parallel (in this version). 
        // Hence, we construct an alternative called set_initial_dilation() that 
        // constructs an L2-projection of $\widetilde{J}=1$ onto the finite element space
        // using tools from step-40.
        set_initial_dilation(solution_n_relevant);

        output_results();
        time.increment();

        while (time.current() < time.end())
        {
            // ...solve the current time step and update total solution vector
            // $\mathbf{\Xi}_{\textrm{n}} = \mathbf{\Xi}_{\textrm{n-1}} +
            // \varDelta \mathbf{\Xi}$...
            solve_nonlinear_timestep(solution_delta);
            {
                TrilinosWrappers::MPI::BlockVector
                tmp(locally_owned_partitioning, locally_relevant_partitioning, mpi_communicator);
                tmp = solution_delta;
                solution_n_relevant += tmp;
            }
            //solution_n_relevant += solution_delta;
            solution_delta = 0.0;

            // ...and plot the results before moving on happily to the next time
            // step:
            output_results();
            time.increment();
        }

        timer.print_summary();
        timer.reset();
    }

// @sect3{Private interface}

// @sect4{Previously called "Threading-building-blocks structures"}

// The first group of private member functions is related to parallelization.
    template <int dim>
    struct Solid<dim>::PerTaskData_ASM
    {
        FullMatrix<double> cell_matrix;
        Vector<double>     cell_rhs;

        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_ASM(const unsigned int dofs_per_cell):
            cell_matrix(dofs_per_cell, dofs_per_cell),
            cell_rhs(dofs_per_cell),
            local_dof_indices(dofs_per_cell)
        {}

        void reset()
        {
            cell_matrix = 0.0;
            cell_rhs = 0.0;
        }
    };

    template <int dim>
    struct Solid<dim>::ScratchData_ASM
    {
        const TrilinosWrappers::MPI::BlockVector &solution_total;

        // Integration helper
        FEValues<dim>      fe_values_ref;
        FEFaceValues<dim>  fe_face_values_ref;

        // Quadrature point solution
        std::vector<Tensor<2, dim> > solution_grads_u_total;
        std::vector<double>          solution_values_p_total;
        std::vector<double>          solution_values_J_total;

        // Shape function values and gradients
        std::vector<std::vector<double> >                   Nx;
        std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;
        std::vector<std::vector<SymmetricTensor<2, dim> > > symm_grad_Nx;

        ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                        const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                        const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face,
                        const TrilinosWrappers::MPI::BlockVector &solution_total):
            solution_total (solution_total),
            fe_values_ref(fe_cell, qf_cell, uf_cell),
            fe_face_values_ref(fe_cell, qf_face, uf_face),
            solution_grads_u_total(qf_cell.size()),
            solution_values_p_total(qf_cell.size()),
            solution_values_J_total(qf_cell.size()),
            Nx (qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
            grad_Nx (qf_cell.size(), std::vector<Tensor<2, dim> >(fe_cell.dofs_per_cell)),
            symm_grad_Nx (qf_cell.size(),
                         std::vector<SymmetricTensor<2, dim> > (fe_cell.dofs_per_cell))
        {}

        ScratchData_ASM(const ScratchData_ASM &rhs):
            solution_total(rhs.solution_total),
            fe_values_ref (rhs.fe_values_ref.get_fe(),
                           rhs.fe_values_ref.get_quadrature(),
                           rhs.fe_values_ref.get_update_flags()),
            fe_face_values_ref (rhs.fe_face_values_ref.get_fe(),
                                rhs.fe_face_values_ref.get_quadrature(),
                                rhs.fe_face_values_ref.get_update_flags()),
            solution_grads_u_total (rhs.solution_grads_u_total),
            solution_values_p_total(rhs.solution_values_p_total),
            solution_values_J_total(rhs.solution_values_J_total),
            Nx(rhs.Nx),
            grad_Nx(rhs.grad_Nx),
            symm_grad_Nx(rhs.symm_grad_Nx)
        {}

        void reset()
        {
            const unsigned int n_q_points = solution_grads_u_total.size();
            const unsigned int n_dofs_per_cell = Nx[0].size();

            Assert(solution_grads_u_total.size() == n_q_points,
                    ExcInternalError());
            Assert(solution_values_p_total.size() == n_q_points,
                    ExcInternalError());
            Assert(solution_values_J_total.size() == n_q_points,
                    ExcInternalError());
            Assert(Nx.size() == n_q_points,
                    ExcInternalError());
            Assert(grad_Nx.size() == n_q_points,
                    ExcInternalError());
            Assert(symm_grad_Nx.size() == n_q_points,
                    ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
                Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
                Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                        ExcInternalError());
                Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                        ExcInternalError());

                solution_grads_u_total[q_point] = 0.0;
                solution_values_p_total[q_point] = 0.0;
                solution_values_J_total[q_point] = 0.0;
                for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
                {
                    Nx[q_point][k] = 0.0;
                    grad_Nx[q_point][k] = 0.0;
                    symm_grad_Nx[q_point][k] = 0.0;
                }
            }
        }
    };

// The ScratchData object will be used to store an alias for the solution
// vector so that we don't have to copy this large data structure. We then
// define a number of vectors to extract the solution values and gradients at
// the quadrature points.
    template <int dim>
    struct Solid<dim>::ScratchData_UQPH
    {
        const TrilinosWrappers::MPI::BlockVector &solution_total;

        // Quadrature point solution
        std::vector<Tensor<2, dim> > solution_grads_u_total;
        std::vector<double>          solution_values_p_total;
        std::vector<double>          solution_values_J_total;

        // Integration helper
        FEValues<dim>      fe_values_ref;

        ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
                         const QGauss<dim> &qf_cell,
                         const UpdateFlags uf_cell,
                         const TrilinosWrappers::MPI::BlockVector &solution_total):
            solution_total(solution_total),
            solution_grads_u_total(qf_cell.size()),
            solution_values_p_total(qf_cell.size()),
            solution_values_J_total(qf_cell.size()),
            fe_values_ref(fe_cell, qf_cell, uf_cell)
        {}

        ScratchData_UQPH(const ScratchData_UQPH &rhs):
            solution_total(rhs.solution_total),
            solution_grads_u_total(rhs.solution_grads_u_total),
            solution_values_p_total(rhs.solution_values_p_total),
            solution_values_J_total(rhs.solution_values_J_total),
            fe_values_ref(rhs.fe_values_ref.get_fe(),
                          rhs.fe_values_ref.get_quadrature(),
                          rhs.fe_values_ref.get_update_flags())
        {}

        void reset()
        {
            const unsigned int n_q_points = solution_grads_u_total.size();
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                solution_grads_u_total[q] = 0.0;
                solution_values_p_total[q] = 0.0;
                solution_values_J_total[q] = 0.0;
            }
        }
    };

// @sect4{Solid::make_grid}

// On to the first of the private member functions. Here we create the
// triangulation of the domain, for which we choose the scaled cube with each
// face given a boundary ID number.  The grid must be refined at least once
// for the indentation problem.
//
// We then determine the volume of the reference configuration and print it
// for comparison:
    template <int dim>
    void Solid<dim>::make_grid()
    {
        GridGenerator::hyper_rectangle(triangulation,
                                       (dim==3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
                                       (dim==3 ? Point<dim>(1.0, 1.0, 1.0) : Point<dim>(1.0, 1.0)),
                                        true);
        GridTools::scale(parameters.scale, triangulation);
        triangulation.refine_global(std::max (1U, parameters.global_refinement));

        // We compute the reference volume of the triangulation. GridTools::volume
        // automatically performs a collective operation when the triangulation is
        // of type parallel::distributed.
        vol_reference = GridTools::volume(triangulation);
        pcout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;

        // Since we wish to apply a Neumann BC to a patch on the top surface, we
        // must find the cell faces in this part of the domain and mark them with
        // a distinct boundary ID number.  The faces we are looking for are on the
        // +y surface and will get boundary ID 6 (zero through five are already
        // used when creating the six faces of the cube domain):
        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    if (cell->face(face)->at_boundary() == true &&
                        cell->face(face)->center()[1] == 1.0 * parameters.scale)
                    {
                        if (dim == 3)
                        {
                            if (cell->face(face)->center()[0] < 0.5 * parameters.scale &&
                                cell->face(face)->center()[2] < 0.5 * parameters.scale)
                                cell->face(face)->set_boundary_id(6);
                        }
                        else
                        {
                            if (cell->face(face)->center()[0] < 0.5 * parameters.scale)
                                cell->face(face)->set_boundary_id(6);
                        }
                    }
                }
            }    
    }

// @sect4{Solid::system_setup}

// Next we describe how the FE system is setup.  We first determine the number
// of components per block. Since the displacement is a vector component, the
// first dim components belong to it, while the next two describe scalar
// pressure and dilatation DOFs.
    template <int dim>
    void Solid<dim>::system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta)
    {
        TimerOutput::Scope t(timer, "Setup system");
        pcout << "Setting up linear system..." << std::endl;

        block_component = std::vector<unsigned int> (n_components, u_block); // Displacement
        block_component[p_component] = p_block; // Pressure
        block_component[J_component] = J_block; // Dilation

        // The DOF handler is then initialised and we renumber the grid in an
        // efficient manner. We also record the number of DOFs per block.
        dof_handler_ref.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler_ref);
        DoFRenumbering::component_wise(dof_handler_ref, block_component);

        // Count DoFs in each block
        dofs_per_block.clear();
        dofs_per_block.resize(n_blocks);
        DoFTools::count_dofs_per_block(dof_handler_ref, dofs_per_block, block_component);

        const unsigned int n_u = dofs_per_block[u_block],
                           n_p = dofs_per_block[p_block],
                           n_J = dofs_per_block[J_block];

        pcout << "  Number of degrees of freedom per block: "
              << "[n_u, n_p, n_J] = ["
              << n_u << ", "
              << n_p << ", "
              << n_J << "]"
              << std::endl;

        // We now define what locally_owned_partitioning and locally_relevant partitioning are
        // We follow step-55 for this
        locally_owned_dofs = dof_handler_ref.locally_owned_dofs();
        locally_owned_partitioning.resize(n_blocks);
        locally_owned_partitioning[u_block] = locally_owned_dofs.get_view(0,n_u);
        locally_owned_partitioning[p_block] = locally_owned_dofs.get_view(n_u, n_u+n_p);
        locally_owned_partitioning[J_block] = locally_owned_dofs.get_view(n_u+n_p, n_u+n_p+n_J);

        DoFTools::extract_locally_relevant_dofs(dof_handler_ref, locally_relevant_dofs);
        locally_relevant_partitioning.resize(n_blocks);
        locally_relevant_partitioning[u_block] = locally_relevant_dofs.get_view(0,n_u);
        locally_relevant_partitioning[p_block] = locally_relevant_dofs.get_view(n_u, n_u+n_p);
        locally_relevant_partitioning[J_block] = locally_relevant_dofs.get_view(n_u+n_p, n_u+n_p+n_J);        

        // Setup the sparsity pattern and tangent matrix (WE FOLLOW STEP 55 FOR THIS)
        {
            tangent_matrix.clear();

            // We optimise the sparsity pattern to reflect the particular
            // structure of the system matrix and prevent unnecessary data 
            // creation for the right-diagonal block components.
            Table<2, DoFTools::Coupling> coupling(n_components, n_components);
            for (unsigned int ii = 0; ii < n_components; ++ii)
                for (unsigned int jj = 0; jj < n_components; ++jj)
                {
                    if ((   (ii <  p_component) && (jj == J_component))
                        || ((ii == J_component) && (jj < p_component))
                        || ((ii == p_component) && (jj == p_component)  ))
                        coupling[ii][jj] = DoFTools::none;
                    else
                        coupling[ii][jj] = DoFTools::always;
                }
            
            TrilinosWrappers::BlockSparsityPattern bsp (locally_owned_partitioning,
                                                        locally_owned_partitioning,
                                                        locally_relevant_partitioning,
                                                        mpi_communicator);
            DoFTools::make_sparsity_pattern (dof_handler_ref, bsp, constraints, false,
                                             this_mpi_process);
            bsp.compress();
            tangent_matrix.reinit(bsp);
        }

        // Next, let us initialize the solution and right hand side vectors. 
        // The solution vector we seek does not only store
        // elements we own, but also ghost entries; on the other hand, the right
        // hand side vector only needs to have the entries the current processor
        // owns since all we will ever do is write into it, never read from it on
        // locally owned cells (of course the linear solvers will read from it,
        // but they do not care about the geometric location of degrees of
        // freedom).
        system_rhs.reinit(locally_owned_partitioning, mpi_communicator);
        solution_n_relevant.reinit(locally_owned_partitioning, locally_relevant_partitioning, mpi_communicator);
        solution_delta.reinit(locally_owned_partitioning, mpi_communicator);

        // Setup quadrature point history
        setup_qph();
    }

// @sect4{Solid::setup_qph}
// The method used to store quadrature information is already described in
// step-18.
//
// Firstly the actual QPH data objects are created. This must be done only
// once the grid is refined to its finest level.    
    template <int dim>
    void Solid<dim>::setup_qph()
    {
        pcout << "    Setting up quadrature point data...\n";
        //quadrature_point_history.initialize(triangulation.begin_active(),
        //                                    triangulation.end(),
        //                                    n_q_points);
        {
            FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
            f_cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.begin_active()),
            f_endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.end());
            quadrature_point_history.initialize(f_cell, f_endc, n_q_points);
        }
        
        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
                const std::vector<std::shared_ptr<PointHistory<dim> > >
                    lqph = quadrature_point_history.get_data(cell);
                Assert(lqph.size() == n_q_points, ExcInternalError());
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                    lqph[q_point]->setup_lqp(parameters);
            }
    }

    template <int dim>
    void Solid<dim>::update_qph(TrilinosWrappers::MPI::BlockVector &solution_delta)
    {
        TimerOutput::Scope t(timer, "Update QPH data");
        const TrilinosWrappers::MPI::BlockVector solution_total(get_solution_total(solution_delta));
        const UpdateFlags uf_UQPH(update_values | update_gradients);
        ScratchData_UQPH scratch_data_uqph(fe, qf_cell, uf_UQPH, solution_total);

        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
                update_qph_one_cell(cell, scratch_data_uqph);
    }

    template <int dim>
    void Solid<dim>::update_qph_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData_UQPH &scratch)
    {
        const std::vector<std::shared_ptr<PointHistory<dim> > > 
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        Assert(scratch.solution_grads_u_total.size() == n_q_points,
            ExcInternalError());
        Assert(scratch.solution_values_p_total.size() == n_q_points,
            ExcInternalError());
        Assert(scratch.solution_values_J_total.size() == n_q_points,
            ExcInternalError());

        scratch.reset();

        // We first need to find the values and gradients at quadrature points
        // inside the current cell and then we update each local QP using the
        // displacement gradient and total pressure and dilatation solution
        // values:
        scratch.fe_values_ref.reinit(cell);
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                           scratch.solution_grads_u_total);
        scratch.fe_values_ref[p_fe].get_function_values(scratch.solution_total,
                                                        scratch.solution_values_p_total);
        scratch.fe_values_ref[J_fe].get_function_values(scratch.solution_total,
                                                        scratch.solution_values_J_total);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            lqph[q_point]->update_values(scratch.solution_grads_u_total[q_point],
                                         scratch.solution_values_p_total[q_point],
                                         scratch.solution_values_J_total[q_point]);
    }

// @sect4{Solid::set_initial_dilation}
// The VectorTools::project() is not available in deal.II v8.5.0 for an MPI-based
// code, so we have to implement it by ourselves. Recall that the main objective
// is to initialize the dilation as J = 1 by projecting this function onto the
// dilation finite element space.
    template <int dim>
    void Solid<dim>::set_initial_dilation(TrilinosWrappers::MPI::BlockVector &solution_n_relevant)
    {
        TimerOutput::Scope t(timer, "Setup initial dilation");
        pcout << "    Setting up initial dilation ..." << std::endl;
        DoFHandler<dim> dof_handler_J(triangulation);
        FE_DGPMonomial<dim> fe_J(parameters.poly_degree - 1);

        IndexSet                         locally_owned_dofs_J;
        IndexSet                         locally_relevant_dofs_J;
        ConstraintMatrix                 constraints_J;
        TrilinosWrappers::SparseMatrix   mass_matrix;
        TrilinosWrappers::MPI::Vector    load_vector;
        TrilinosWrappers::MPI::Vector    J_local;       

        // Setup system
        dof_handler_J.distribute_dofs(fe_J);
        locally_owned_dofs_J = dof_handler_J.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler_J, locally_relevant_dofs_J);
        
        J_local.reinit(locally_owned_dofs_J, locally_relevant_dofs_J, mpi_communicator);
        load_vector.reinit(locally_owned_dofs_J, mpi_communicator);

        constraints_J.clear();
        constraints_J.reinit(locally_relevant_dofs_J);
        constraints_J.close();

        DynamicSparsityPattern dsp_J(locally_relevant_dofs_J);
        DoFTools::make_sparsity_pattern(dof_handler_J, dsp_J, constraints_J, false);
        SparsityTools::distribute_sparsity_pattern(dsp_J,
                                                   dof_handler_J.n_locally_owned_dofs_per_processor(),
                                                   mpi_communicator,
                                                   locally_relevant_dofs_J);
        mass_matrix.reinit(locally_owned_dofs_J, locally_owned_dofs_J, dsp_J, mpi_communicator);

        // Assemble system
        const QGauss<dim> quad_formula(parameters.quad_order);
        FEValues<dim> fe_values_J(fe_J, quad_formula, 
                                  update_values | update_quadrature_points | update_JxW_values);
        const unsigned int dofs_per_cell_J = fe_J.dofs_per_cell;
        const unsigned int n_q_points_J = quad_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell_J, dofs_per_cell_J);
        Vector<double> cell_rhs(dofs_per_cell_J);
        std::vector<types::global_dof_index> local_dof_indices_J(dofs_per_cell_J);

        for (const auto &cell : dof_handler_J.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                cell_matrix = 0;
                cell_rhs = 0;
                fe_values_J.reinit(cell);

                for (unsigned int q_point = 0; q_point < n_q_points_J; ++q_point)
                {
                    for (unsigned int i = 0; i < dofs_per_cell_J; ++i)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell_J; ++j)
                        {
                            cell_matrix(i,j) += (fe_values_J.shape_value(i,q_point) *
                                                 fe_values_J.shape_value(j,q_point) *
                                                 fe_values_J.JxW(q_point));
                            
                            cell_rhs(i) += (1 * 
                                            fe_values_J.shape_value(i,q_point) * 
                                            fe_values_J.JxW(q_point));
                        }
                    }
                }
                cell->get_dof_indices(local_dof_indices_J);
                constraints_J.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices_J,
                                                         mass_matrix, load_vector);
            }   

        // Notice that the assembling above is just a local operation. So, to
        // form the "global" linear system, a synchronization between all
        // processors is needed. This could be done by invoking the function
        // compress(). See @ref GlossCompress  "Compressing distributed objects"
        // for more information on what is compress() designed to do. 
        mass_matrix.compress(VectorOperation::add);
        load_vector.compress(VectorOperation::add);

        // Solve
        TrilinosWrappers::MPI::Vector J_distributed(locally_owned_dofs_J, mpi_communicator);
        SolverControl solver_control(dof_handler_J.n_dofs(), 1e-12);
        dealii::SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

        TrilinosWrappers::PreconditionJacobi preconditioner;
        preconditioner.initialize(mass_matrix);
        solver.solve(mass_matrix, J_distributed, load_vector, preconditioner);
        constraints_J.distribute(J_distributed);

        // We have computed the J=1 in the finite element space. We now transfer
        // this quantity to the solution_n_relevant vector.
        solution_n_relevant.block(J_block) = J_distributed;
        
        // We destroy the dof_handler_J object and leave the subsection.
        dof_handler_J.clear();  
    }

// @sect4{Solid::solve_nonlinear_timestep}

// The next function is the driver method for the Newton-Raphson scheme. At
// its top we create a new vector to store the current Newton update step,
// reset the error storage objects and print solver header.
    template <int dim>
    void Solid<dim>::solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta)
    {
        pcout << std::endl
              << "Timestep " << time.get_timestep() << " @ "
              << time.current() << "s of "
              << time.end() << "s" << std::endl;

        TrilinosWrappers::MPI::BlockVector newton_update;
        newton_update.reinit(locally_owned_partitioning, mpi_communicator);

        error_residual.reset();
        error_residual_0.reset();
        error_residual_norm.reset();
        error_update.reset();
        error_update_0.reset();
        error_update_norm.reset();
        
        print_conv_header(); 

        unsigned int newton_iteration = 0;
        for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
        {
            pcout << " " << std::setw(2) << newton_iteration << " " << std::flush;
            make_constraints(newton_iteration);
            assemble_system(solution_delta);
            get_error_residual(error_residual);

            if (newton_iteration == 0)
                error_residual_0 = error_residual;

            // We can now determine the normalised residual error and check for
            // solution convergence:
            error_residual_norm = error_residual;
            error_residual_norm.normalise(error_residual_0);

            if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u
                                     && error_residual_norm.u <= parameters.tol_f)
            {
                pcout << " CONVERGED! " << std::endl;
                print_conv_footer();
                break;
            }

            const std::pair<unsigned int, double>
                lin_solver_output = solve_linear_system(newton_update);

            get_error_update(newton_update, error_update);
            if (newton_iteration == 0)
                error_update_0 = error_update;

            // We can now determine the normalised Newton update error, and
            // perform the actual update of the solution increment for the current
            // time step, update all quadrature point information pertaining to
            // this new displacement and stress state and continue iterating:
            error_update_norm = error_update;
            error_update_norm.normalise(error_update_0);
            solution_delta += newton_update;
            update_qph(solution_delta);
            newton_update = 0.0;

            pcout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << lin_solver_output.first << "  "
              << lin_solver_output.second << "  " << error_residual_norm.norm
              << "  " << error_residual_norm.u << "  "
              << error_residual_norm.p << "  " << error_residual_norm.J
              << "  " << error_update_norm.norm << "  " << error_update_norm.u
              << "  " << error_update_norm.p << "  " << error_update_norm.J
              << "  " << std::endl;
        }
        AssertThrow (newton_iteration <= parameters.max_iterations_NR,
                     ExcMessage("No convergence in nonlinear solver!"));
    }

// @sect4{Solid::print_conv_header and Solid::print_conv_footer}

// This program prints out data in a nice table that is updated
// on a per-iteration basis. The next two functions set up the table
// header and footer:
    template <int dim>
    void Solid<dim>::print_conv_header()
    {
        pcout << std::string(132,'_') << std::endl;
        pcout << "     SOLVER STEP       "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     RES_P      RES_J     NU_NORM     "
              << " NU_U       NU_P       NU_J " << std::endl;
        pcout << std::string(132,'_') << std::endl;
    }
    template <int dim>
    void Solid<dim>::print_conv_footer()
    {
        pcout << std::string(132,'_') << std::endl;
        const std::pair<double,double > error_dil = get_error_dilation();
        pcout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
              << "Dilatation:\t" << error_dil.first << std::endl
              << "v / V_0:\t" << error_dil.second *vol_reference << " / " << vol_reference
              << " = " << error_dil.second << std::endl;
    }
    
    // @sect4{Solid::get_error_dilation}

    // Calculate the volume of the domain in the spatial configuration
    template <int dim>
    double Solid<dim>::compute_vol_current() const
    {
        double vol_current = 0.0;

        FEValues<dim> fe_values_ref(fe, qf_cell, update_JxW_values);

        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                fe_values_ref.reinit(cell);
                // In contrast to that which was previously called for,
                // in this instance the quadrature point data is specifically
                // non-modifiable since we will only be accessing data.
                // We ensure that the right get_data function is called by
                // marking this update function as constant.
                const std::vector<std::shared_ptr<const PointHistory<dim> > > 
                    lqph = quadrature_point_history.get_data(cell);
                Assert(lqph.size() == n_q_points, ExcInternalError());

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const double det_F_qp = lqph[q_point]->get_det_F();
                    const double JxW = fe_values_ref.JxW(q_point);

                    vol_current += det_F_qp * JxW;
                }
            }

        // Reduce the result
        vol_current = Utilities::MPI::sum(vol_current, mpi_communicator);
        Assert(vol_current > 0.0, ExcInternalError());
        return vol_current;
    }

    // Calculate how well the dilatation $\widetilde{J}$ agrees with $J :=
    // \textrm{det}\ \mathbf{F}$ from the $L^2$ error $ \bigl[ \int_{\Omega_0} {[ J
    // - \widetilde{J}]}^{2}\textrm{d}V \bigr]^{1/2}$.
    // We also return the ratio of the current volume of the
    // domain to the reference volume. This is of interest for incompressible
    // media where we want to check how well the isochoric constraint has been
    // enforced.
    template <int dim>
    std::pair<double, double> Solid<dim>::get_error_dilation() const
    {
        double dil_L2_error = 0.0;

        FEValues<dim> fe_values_ref(fe, qf_cell, update_JxW_values);

        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                fe_values_ref.reinit(cell);

                const std::vector<std::shared_ptr<const PointHistory<dim> > > 
                    lqph = quadrature_point_history.get_data(cell);
                Assert(lqph.size() == n_q_points, ExcInternalError());

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const double det_F_qp = lqph[q_point]->get_det_F();
                    const double J_tilde_qp = lqph[q_point]->get_J_tilde();
                    const double the_error_qp_squared = std::pow((det_F_qp - J_tilde_qp), 2);
                    const double JxW = fe_values_ref.JxW(q_point);

                    dil_L2_error += the_error_qp_squared * JxW;
                }
            }

        dil_L2_error = Utilities::MPI::sum(dil_L2_error, mpi_communicator);
        return std::make_pair(std::sqrt(dil_L2_error), compute_vol_current() / vol_reference);
    }

// @sect4{Solid::get_error_residual}

// Determine the true residual error for the problem.  That is, determine the
// error in the residual for the unconstrained degrees of freedom.  Note that to
// do so, we need to ignore constrained DOFs by setting the residual in these
// vector components to zero.
    template <int dim>
    void Solid<dim>::get_error_residual(Errors &error_residual)
    {
        // Construct a residual vector that has the values for all of its
        // constrained DoFs set to zero.
        TrilinosWrappers::MPI::BlockVector error_res (system_rhs);
        constraints.set_zero(error_res);
        error_residual.norm = error_res.l2_norm();
        error_residual.u = error_res.block(u_block).l2_norm();
        error_residual.p = error_res.block(p_block).l2_norm();
        error_residual.J = error_res.block(J_block).l2_norm();
    }

// @sect4{Solid::get_error_update}

// Determine the true Newton update error for the problem
    template <int dim>
    void Solid<dim>::get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update,
                                      Errors &error_update)
    {
        // Construct a update vector that has the values for all of its
        // constrained DoFs set to zero.
        TrilinosWrappers::MPI::BlockVector error_ud (newton_update);
        constraints.set_zero(error_ud);
        error_update.norm = error_ud.l2_norm();
        error_update.u = error_ud.block(u_block).l2_norm();
        error_update.p = error_ud.block(p_block).l2_norm();
        error_update.J = error_ud.block(J_block).l2_norm();
    }

// @sect4{Solid::get_solution_total}

// This function provides the total solution, which is valid at any Newton step.
// This is required as, to reduce computational error, the total solution is
// only updated at the end of the timestep.
    template <int dim>
    TrilinosWrappers::MPI::BlockVector
    Solid<dim>::get_solution_total(const TrilinosWrappers::MPI::BlockVector &solution_delta) const
    {
        TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                          locally_relevant_partitioning,
                                                          mpi_communicator,
                                                          /*vector_writable=*/false);
        TrilinosWrappers::MPI::BlockVector tmp(solution_total);
        solution_total = solution_n_relevant;
        tmp = solution_delta;
        solution_total += tmp;
        return solution_total;
    }

// @sect4{Solid::assemble_system}

    template <int dim>
    void Solid<dim>::assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_delta)
    {
        TimerOutput::Scope t(timer, "Assemble system");
        pcout << " ASM_SYS " << std::flush;
        tangent_matrix = 0.0;
        system_rhs = 0.0;

        const TrilinosWrappers::MPI::BlockVector 
            solution_total(get_solution_total(solution_delta));
        
        const UpdateFlags uf_cell(update_values | update_gradients      | update_JxW_values);
        const UpdateFlags uf_face(update_values | update_normal_vectors | update_JxW_values);

        PerTaskData_ASM per_task_data(dofs_per_cell);
        ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face, solution_total);

        for (const auto &cell : dof_handler_ref.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
                assemble_system_one_cell(cell, scratch_data, per_task_data);
                copy_local_to_global_system(per_task_data);
            }
        
        tangent_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }

// Since the assembly of the tangent_matrix and system_rhs are performed in the same function,
// we can call the constraints.distribute_local_to_global to do the work in a single command.
    template <int dim>
    void Solid<dim>::copy_local_to_global_system(const PerTaskData_ASM &data)
    {
        constraints.distribute_local_to_global(data.cell_matrix, data.cell_rhs,
                                               data.local_dof_indices,
                                               tangent_matrix, system_rhs);
    }

// @sect4{Solid::assemble_system_one_cell} +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// The assemble of the local matrix and right-hand side is probably the most involved of all the
// functions so far. Here is where the physics of the problem can be found.
    template <int dim>
    void Solid<dim>::assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM &scratch,
                                              PerTaskData_ASM &data) const
    {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError()); // Sanity check

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);
        const std::vector<std::shared_ptr<const PointHistory<dim> > > lqph =
        quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        // Update quadrature point solution
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                           scratch.solution_grads_u_total);
        scratch.fe_values_ref[p_fe].get_function_values(scratch.solution_total,
                                                        scratch.solution_values_p_total);
        scratch.fe_values_ref[J_fe].get_function_values(scratch.solution_total,
                                                        scratch.solution_values_J_total);

        // Update shape functions and their gradients (push-forward)
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim> F_inv = lqph[q_point]->get_F_inv();
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_block)
                {
                    scratch.grad_Nx[q_point][k] 
                        = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                    scratch.symm_grad_Nx[q_point][k] 
                        = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else if (k_group == p_block)
                    scratch.Nx[q_point][k] = scratch.fe_values_ref[p_fe].value(k, q_point);
                else if (k_group == J_block)
                    scratch.Nx[q_point][k] = scratch.fe_values_ref[J_fe].value(k, q_point);
                else
                    Assert(k_group <= J_block, ExcInternalError());
            }
        }

        // Now we build the local cell stiffness matrix and local right-hand side. 
        // Since the global and local system matrices are symmetric, we can exploit 
        // this property by building only the lower half of the local matrix and 
        // copying the values to the upper half.  So we only assemble half of the
        // $\mathsf{\mathbf{k}}_{uu}$, $\mathsf{\mathbf{k}}_{\widetilde{p}
        // \widetilde{p}} = \mathbf{0}$, $\mathsf{\mathbf{k}}_{\widetilde{J}
        // \widetilde{J}}$ blocks, while the whole
        // $\mathsf{\mathbf{k}}_{\widetilde{p} \widetilde{J}}$,
        // $\mathsf{\mathbf{k}}_{u \widetilde{J}} = \mathbf{0}$,
        // $\mathsf{\mathbf{k}}_{u \widetilde{p}}$ blocks are built.
        //
        // In doing so, we first extract some configuration dependent variables
        // from our quadrature history objects for the current quadrature point.
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const SymmetricTensor<2, dim> tau = lqph[q_point]->get_tau(); 
            const Tensor<2, dim> tau_ns(tau);
            const SymmetricTensor<4, dim> Jc  = lqph[q_point]->get_Jc();
            const double dPsi_vol_dJ          = lqph[q_point]->get_dPsi_vol_dJ();
            const double d2Psi_vol_dJ2        = lqph[q_point]->get_d2Psi_vol_dJ2();
            const double det_F                = lqph[q_point]->get_det_F();
            const double p_tilde              = lqph[q_point]->get_p_tilde();
            const double J_tilde              = lqph[q_point]->get_J_tilde();
            Assert(det_F > 0, ExcInternalError());

            // Next we define some aliases to make the assembly process easier to
            // follow
            const std::vector<double>
                &N = scratch.Nx[q_point];
            const std::vector<SymmetricTensor<2, dim> >
                &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim> >
                &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                // LOCAL MATRIX
                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                    // This is the $\mathsf{\mathbf{k}}_{uu}$
                    // contribution. It comprises a material contribution, and a
                    // geometrical stress contribution which is only added along
                    // the local matrix diagonals:
                    if ((i_group == j_group) && (i_group == u_block))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                                * symm_grad_Nx[j] * JxW;
                        if (component_i == component_j) // geometrical stress contribution
                            data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                    * grad_Nx[j][component_j] * JxW;
                    }
                    // Next is the $\mathsf{\mathbf{k}}_{ \widetilde{p} u}$ contribution
                    else if ((i_group == p_block) && (j_group == u_block))
                    {
                        data.cell_matrix(i, j) += N[i] * det_F
                                                * (symm_grad_Nx[j]
                                                    * Physics::Elasticity::StandardTensors<dim>::I)
                                                * JxW;
                    }
                    // and lastly the $\mathsf{\mathbf{k}}_{ \widetilde{J} \widetilde{p}}$
                    // and $\mathsf{\mathbf{k}}_{ \widetilde{J} \widetilde{J}}$
                    // contributions:
                    else if ((i_group == J_block) && (j_group == p_block))
                        data.cell_matrix(i, j) -= N[i] * N[j] * JxW;
                    else if ((i_group == j_group) && (i_group == J_block))
                        data.cell_matrix(i, j) += N[i] * d2Psi_vol_dJ2 * N[j] * JxW;
                    else
                        Assert((i_group <= J_block) && (j_group <= J_block), ExcInternalError());
                }

                // LOCAL RIGHT-HAND SIDE
                if (i_group == u_block)
                    data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
                else if (i_group == p_block)
                    data.cell_rhs(i) -= N[i] * (det_F - J_tilde) * JxW;
                else if (i_group == J_block)
                    data.cell_rhs(i) -= N[i] * (dPsi_vol_dJ - p_tilde) * JxW;
                else
                    Assert(i_group <= J_block, ExcInternalError());
            }
        }

        // We need to copy the lower half of the local matrix into the upper half:
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);

        // Next we assemble the Neumann contribution. We first check to see if the
        // cell face exists on a boundary on which a traction is applied and add
        // the contribution if this is the case.
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 6)
            {
                scratch.fe_face_values_ref.reinit(cell, face);
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                {
                    const Tensor<1, dim> &N = scratch.fe_face_values_ref.normal_vector(f_q_point);

                    // Using the face normal at this quadrature point we specify the
                    // traction in reference configuration. For this problem, a
                    // defined pressure is applied in the reference configuration.
                    // The direction of the applied traction is assumed not to
                    // evolve with the deformation of the domain. The traction is
                    // defined using the first Piola-Kirchhoff stress is simply
                    // $\mathbf{t} = \mathbf{P}\mathbf{N} = [p_0 \mathbf{I}]
                    // \mathbf{N} = p_0 \mathbf{N}$ We use the time variable to
                    // linearly ramp up the pressure load.
                    //
                    // Note that the contributions to the right hand side vector we
                    // compute here only exist in the displacement components of the
                    // vector.
                    static const double  p0        = -4.0
                                                    /
                                                    (parameters.scale * parameters.scale);
                    const double         time_ramp = (time.current() / time.end());
                    const double         pressure  = p0 * parameters.p_p0 * time_ramp;
                    const Tensor<1, dim> traction  = pressure * N;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int i_group =
                            fe.system_to_base_index(i).first.first;

                        if (i_group == u_block)
                        {
                            const unsigned int component_i =
                                fe.system_to_component_index(i).first;
                            const double Ni =
                                scratch.fe_face_values_ref.shape_value(i,
                                                                    f_q_point);
                            const double JxW = scratch.fe_face_values_ref.JxW(
                                                f_q_point);

                            data.cell_rhs(i) += (Ni * traction[component_i])
                                                * JxW;
                        }
                    }
                }
            }
    }

// @sect4{Solid::make_constraints}
// The constraints for this problem are simple to describe.
// However, since we are dealing with an iterative Newton method,
// it should be noted that any displacement constraints should only
// be specified at the zeroth iteration and subsequently no
// additional contributions are to be made since the constraints
// are already exactly satisfied.
    template <int dim>
    void Solid<dim>::make_constraints(const int &it_nr)
    {
        pcout << " CST " << std::flush;

        // Since the constraints are different at different Newton iterations, we
        // need to clear the constraints matrix and completely rebuild
        // it. However, after the first iteration, the constraints remain the same
        // and we can simply skip the rebuilding step if we do not clear it.
        if (it_nr > 1)
            return;
        constraints.clear();
        const bool apply_dirichlet_bc = (it_nr == 0);

        // The boundary conditions for the indentation problem are as follows: On
        // the -x, -y and -z faces (IDs 0,2,4) we set up a symmetry condition to
        // allow only planar movement while the +x and +z faces (IDs 1,5) are
        // traction free. In this contrived problem, part of the +y face (ID 3) is
        // set to have no motion in the x- and z-component. Finally, as described
        // earlier, the other part of the +y face has an the applied pressure but
        // is also constrained in the x- and z-directions.
        //
        // In the following, we will have to tell the function interpolation
        // boundary values which components of the solution vector should be
        // constrained (i.e., whether it's the x-, y-, z-displacements or
        // combinations thereof). This is done using ComponentMask objects (see
        // @ref GlossComponentMask) which we can get from the finite element if we
        // provide it with an extractor object for the component we wish to
        // select. To this end we first set up such extractor objects and later
        // use it when generating the relevant component masks:
        const FEValuesExtractors::Scalar x_displacement(0);
        const FEValuesExtractors::Scalar y_displacement(1);

        {
            const int boundary_id = 0;
            if (apply_dirichlet_bc == true)
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(x_displacement));
            else
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(x_displacement));
        }
        {
            const int boundary_id = 2;
            if (apply_dirichlet_bc == true)
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(y_displacement));
            else
                VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                        boundary_id,
                                                        ZeroFunction<dim>(n_components),
                                                        constraints,
                                                        fe.component_mask(y_displacement));
        }

        if (dim == 3)
        {
            const FEValuesExtractors::Scalar z_displacement(2);

            {
                const int boundary_id = 3;
                if (apply_dirichlet_bc == true)
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)
                                                            |
                                                            fe.component_mask(z_displacement)));
                else
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)
                                                            |
                                                            fe.component_mask(z_displacement)));
            }
            {
                const int boundary_id = 4;
                if (apply_dirichlet_bc == true)
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            fe.component_mask(z_displacement));
                else
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            fe.component_mask(z_displacement));
            }

            {
                const int boundary_id = 6;
                if (apply_dirichlet_bc == true)
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)
                                                            |
                                                            fe.component_mask(z_displacement)));
                else
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)
                                                            |
                                                            fe.component_mask(z_displacement)));
            }
        }
        else
        {
            {
                const int boundary_id = 3;
                if (apply_dirichlet_bc == true)
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)));
                else
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)));
            }
            {
                const int boundary_id = 6;
                if (apply_dirichlet_bc == true)
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)));
                else
                    VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                            boundary_id,
                                                            ZeroFunction<dim>(n_components),
                                                            constraints,
                                                            (fe.component_mask(x_displacement)));
            }
        }

        constraints.close();
    }

// @sect4{Solid::solve_linear_system}
// We now have all of the necessary components to use one of two possible
// methods to solve the linearised system. The full block
// system can be solved by performing condensation on a global level.
// Below we implement this approach.
    template <int dim>
    std::pair<unsigned int, double> 
    Solid<dim>::solve_linear_system(TrilinosWrappers::MPI::BlockVector &newton_update)
    {
        unsigned int lin_it = 0;
        double lin_res = 0.0;

        TimerOutput::Scope t(timer, "Linear solver");
        pcout << " SLV " << std::flush;

        // For ease of later use, we define some aliases for
        // blocks in the RHS vector
        const TrilinosWrappers::MPI::Vector &f_u = system_rhs.block(u_block);
        const TrilinosWrappers::MPI::Vector &f_p = system_rhs.block(p_block);
        const TrilinosWrappers::MPI::Vector &f_J = system_rhs.block(J_block);

        // ... and for blocks in the Newton update vector.
        TrilinosWrappers::MPI::Vector &d_u = newton_update.block(u_block);
        TrilinosWrappers::MPI::Vector &d_p = newton_update.block(p_block);
        TrilinosWrappers::MPI::Vector &d_J = newton_update.block(J_block);

        // We next define some linear operators for the tangent matrix sub-blocks
        // We will exploit the symmetry of the system, so not all blocks
        // are required.
        const auto K_uu = linear_operator<TrilinosWrappers::MPI::Vector>(tangent_matrix.block(u_block, u_block));
        const auto K_up = linear_operator<TrilinosWrappers::MPI::Vector>(tangent_matrix.block(u_block, p_block));
        const auto K_pu = linear_operator<TrilinosWrappers::MPI::Vector>(tangent_matrix.block(p_block, u_block));
        const auto K_Jp = linear_operator<TrilinosWrappers::MPI::Vector>(tangent_matrix.block(J_block, p_block));
        const auto K_JJ = linear_operator<TrilinosWrappers::MPI::Vector>(tangent_matrix.block(J_block, J_block));

        // We then construct a LinearOperator that represents the inverse of (square block)
        // $\mathsf{\mathbf{K}}_{\widetilde{J}\widetilde{p}}$. Since it is diagonal (or,
        // when a higher order ansatz it used, nearly diagonal), a Jacobi preconditioner
        // is suitable.
        TrilinosWrappers::PreconditionJacobi preconditioner_K_Jp_inv;
        preconditioner_K_Jp_inv.initialize(tangent_matrix.block(J_block, p_block),
                                           TrilinosWrappers::PreconditionJacobi::AdditionalData());
        ReductionControl 
        solver_control_K_Jp_inv (static_cast<unsigned int>(tangent_matrix.block(J_block, p_block).m() 
                                 * parameters.max_iterations_lin), 1.0e-30, 1e-6);
        dealii::SolverCG<TrilinosWrappers::MPI::Vector> solver_K_Jp_inv (solver_control_K_Jp_inv);

        const auto K_Jp_inv = inverse_operator(K_Jp,
                                               solver_K_Jp_inv,
                                               preconditioner_K_Jp_inv);

        // Now we can construct that transpose of $\mathsf{\mathbf{K}}_{\widetilde{J}\widetilde{p}}^{-1}$
        // and a linear operator that represents the condensed operations
        // $\overline{\mathsf{\mathbf{K}}}$ and
        // $\overline{\overline{\mathsf{\mathbf{K}}}}$ and the final augmented matrix
        // $\mathsf{\mathbf{K}}_{\textrm{con}}$.
        // Note that the schur_complement() operator could also be of use here, but
        // for clarity and the purpose of demonstrating the similarities between the
        // formulation and implementation of the linear solution scheme, we will perform
        // these operations manually.
        const auto K_pJ_inv     = transpose_operator(K_Jp_inv);
        const auto K_pp_bar     = K_Jp_inv * K_JJ * K_pJ_inv;
        const auto K_uu_bar_bar = K_up * K_pp_bar * K_pu;
        const auto K_uu_con     = K_uu + K_uu_bar_bar;

        // Lastly, we define an operator for inverse of augmented stiffness matrix,
        // namely $\mathsf{\mathbf{K}}_{\textrm{con}}^{-1}$.
        TrilinosWrappers::PreconditionAMG preconditioner_K_con_inv;
        preconditioner_K_con_inv.initialize(tangent_matrix.block(u_block, u_block),
            TrilinosWrappers::PreconditionAMG::AdditionalData(true /*elliptic*/,
                                               (parameters.poly_degree > 1 /*higher_order_elements*/)));
        ReductionControl solver_control_K_con_inv (
            static_cast<unsigned int>(tangent_matrix.block(u_block, u_block).m() 
                                        * parameters.max_iterations_lin),
                                      1.0e-30, parameters.tol_lin);
        dealii::SolverSelector<TrilinosWrappers::MPI::Vector> solver_K_con_inv;
        solver_K_con_inv.select(parameters.type_lin);
        solver_K_con_inv.set_control(solver_control_K_con_inv);
        const auto K_uu_con_inv = inverse_operator(K_uu_con,
                                                   solver_K_con_inv,
                                                   preconditioner_K_con_inv);

        d_u     = K_uu_con_inv*(f_u - K_up*(K_Jp_inv*f_J - K_pp_bar*f_p));
        lin_it  = solver_control_K_con_inv.last_step();
        lin_res = solver_control_K_con_inv.last_value();

        d_J = K_pJ_inv*(f_p - K_pu*d_u);
        d_p = K_Jp_inv*(f_J - K_JJ*d_J);

        constraints.distribute(newton_update);
        return std::make_pair(lin_it, lin_res);
    }

// @sect4{Solid::output_results}
// Solid::FilteredDataOut
    template<int dim, class DH=DoFHandler<dim> >
    class FilteredDataOut : public DataOut<dim, DH>
    {
    public:
        FilteredDataOut (const unsigned int subdomain_id):
            subdomain_id (subdomain_id)
        {}

        virtual ~FilteredDataOut() {}

        virtual typename DataOut<dim, DH>::cell_iterator
        first_cell ()
        {
            auto cell = this->dofs->begin_active();
            while ((cell != this->dofs->end()) && (cell->subdomain_id() != subdomain_id))
                ++cell;
            return cell;
        }

        virtual typename DataOut<dim, DH>::cell_iterator
        next_cell (const typename DataOut<dim, DH>::cell_iterator &old_cell)
        {
            if (old_cell != this->dofs->end())
            {
                const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
                return
                    ++(FilteredIterator
                    <typename DataOut<dim, DH>::cell_iterator>
                    (predicate,old_cell));
            }
            else
                return old_cell;
        }

    private:
        const unsigned int subdomain_id;
    };

    template <int dim>
    void Solid<dim>::output_results() const
    {
        TimerOutput::Scope t(timer, "Output results");
        TrilinosWrappers::MPI::BlockVector solution_total (locally_owned_partitioning,
                                                           locally_relevant_partitioning,
                                                           mpi_communicator,
                                                           /*vector_writable = */ false); 
        TrilinosWrappers::MPI::BlockVector residual (locally_owned_partitioning,
                                                     locally_relevant_partitioning,
                                                     mpi_communicator,
                                                     /*vector_writable = */ false);
        solution_total = solution_n_relevant;
        residual = system_rhs;
        residual *= -1.0;

        // --- Additional data ---
        Vector<double> material_id;
        Vector<double> polynomial_order;
        material_id.reinit(triangulation.n_active_cells());
        polynomial_order.reinit(triangulation.n_active_cells());
        std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());

        FilteredDataOut<dim> data_out(this_mpi_process);
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                    DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        GridTools::get_subdomain_association (triangulation, partition_int);

        // Can't use filtered iterators here because the cell
        // count "c" is incorrect for the parallel case
        unsigned int c = 0;
        typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_ref.begin_active(),
            endc = dof_handler_ref.end();
        for (; cell!=endc; ++cell, ++c)
        {
            if (cell->subdomain_id() != this_mpi_process) continue;
            material_id(c) = static_cast<int>(cell->material_id());
        }

        std::vector<std::string> solution_name(n_components, "solution_");
        std::vector<std::string> residual_name(n_components, "residual_");
        for (unsigned int c=0; c<n_components; ++c)
        {
            if (block_component[c] == u_block)
            {
                solution_name[c] += "u";
                residual_name[c] += "u";
            }
            else if (block_component[c] == p_block)
            {
                solution_name[c] += "p";
                residual_name[c] += "p";
            }
            else if (block_component[c] == J_block)
            {
                solution_name[c] += "J";
                residual_name[c] += "J";
            }
            else
            {
                Assert(c <= J_block, ExcInternalError());
            }
        }

        data_out.attach_dof_handler(dof_handler_ref);
        data_out.add_data_vector(solution_total,
                                solution_name,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
        data_out.add_data_vector(residual,
                                residual_name,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
        const Vector<double> partitioning(partition_int.begin(),
                                        partition_int.end());
        data_out.add_data_vector (material_id, "material_id");
        data_out.add_data_vector (partitioning, "partitioning");

        // Since we are dealing with a large deformation problem, it would be nice
        // to display the result on a displaced grid!  The MappingQEulerian class
        // linked with the DataOut class provides an interface through which this
        // can be achieved without physically moving the grid points in the
        // Triangulation object ourselves.  We first need to copy the solution to
        // a temporary vector and then create the Eulerian mapping. We also
        // specify the polynomial degree to the DataOut object in order to produce
        // a more refined output data set when higher order polynomials are used.
        MappingQEulerian<dim, TrilinosWrappers::MPI::BlockVector> q_mapping(degree, dof_handler_ref, solution_n_relevant);
        
        //data_out.build_patches(degree);
        data_out.build_patches(q_mapping, degree);

        struct Filename
        {
            static std::string get_filename_vtu (unsigned int process,
                                                 unsigned int timestep,
                                                 const unsigned int n_digits = 4)
            {
                std::ostringstream filename_vtu;
                filename_vtu
                    << "solution-"
                    << (std::to_string(dim) + "d")
                    << "."
                    << Utilities::int_to_string (process, n_digits)
                    << "."
                    << Utilities::int_to_string(timestep, n_digits)
                    << ".vtu";
                return filename_vtu.str();
            }

            static std::string get_filename_pvtu (unsigned int timestep,
                                                    const unsigned int n_digits = 4)
            {
                std::ostringstream filename_vtu;
                filename_vtu
                    << "solution-"
                    << (std::to_string(dim) + "d")
                    << "."
                    << Utilities::int_to_string(timestep, n_digits)
                    << ".pvtu";
                return filename_vtu.str();
            }

            static std::string get_filename_pvd (void)
            {
                std::ostringstream filename_vtu;
                filename_vtu
                    << "solution-"
                    << (std::to_string(dim) + "d")
                    << ".pvd";
                return filename_vtu.str();
            }
        };

        // Write out main data file
        const unsigned int timestep = time.get_timestep();
        const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process, timestep);
        std::ofstream output(filename_vtu.c_str());
        data_out.write_vtu(output);

        // Collection of files written in parallel
        // This next set of steps should only be performed
        // by master process
        if (this_mpi_process == 0)
        {
            // List of all files written out at this timestep by all processors
            std::vector<std::string> parallel_filenames_vtu;
            for (unsigned int p=0; p < n_mpi_processes; ++p)
                {
                    parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, timestep));
                }

            const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
            std::ofstream pvtu_master(filename_pvtu.c_str());
            data_out.write_pvtu_record(pvtu_master,
                                        parallel_filenames_vtu);

            // Time dependent data master file
            static std::vector<std::pair<double,std::string> > time_and_name_history;
            time_and_name_history.push_back (std::make_pair (time.current(),
                                                             filename_pvtu));
            const std::string filename_pvd (Filename::get_filename_pvd());
            std::ofstream pvd_output (filename_pvd.c_str());
            DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
        }
    }

}// END OF NAMESPACE Step44

// @sect3{Main function}
// Lastly we provide the main driver function.
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
