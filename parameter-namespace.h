#include <deal.II/base/parameter_handler.h>

using namespace dealii;
namespace Parameters
{
    // @sect3{Run-time parameters}
    //
    // There are several parameters that can be set in the code so we set up a
    // ParameterHandler object to read in the choices at run-time.

    struct FESystem{
        unsigned int poly_degree;
        unsigned int quad_order;
        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Geometry} =======================================================================================

    // Make adjustments to the problem geometry and the applied load.  Since the
    // problem modelled here is quite specific, the load scale can be altered to
    // specific values to compare with the results given in the literature.
    struct Geometry
    {
        unsigned int global_refinement;
        double       scale;
        double       p_p0;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Materials} =============================================================================

    // We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
    // neo-Hookean material.
    struct Materials
    {
        double nu;
        double mu;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Linear solver} ================================================================================

    // Next, we choose both solver and preconditioner settings.  The use of an
    // effective preconditioner is critical to ensure convergence when a large
    // nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
        std::string type_lin;
        double      tol_lin;
        double      max_iterations_lin;
        bool        use_static_condensation;
        std::string preconditioner_type;
        double      preconditioner_relaxation;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Nonlinear solver} ================================================================

    // A Newton-Raphson scheme is used to solve the nonlinear system of governing
    // equations.  We now define the tolerances and the maximum number of
    // iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
        unsigned int max_iterations_NR;
        double       tol_f;
        double       tol_u;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{Time} ===============================================================================

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
        double delta_t;
        double end_time;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };

    // @sect4{All parameters} ======================================================================

    // Finally we consolidate all of the above structures into a single container
    // that holds all of our run-time selections.
    struct AllParameters : 
        public FESystem,
        public Geometry,
        public Materials,
        public LinearSolver,
        public NonlinearSolver,
        public Time
    {
        AllParameters(const std::string &input_file);

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
    };
}