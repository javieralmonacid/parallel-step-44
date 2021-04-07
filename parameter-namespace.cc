#include <deal.II/base/parameter_handler.h>
#include "parameter-namespace.h"

//using namespace dealii;
namespace Parameters
{
    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry} =======================================================================================

    void Geometry::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Geometry");
        {
            prm.declare_entry("Global refinement", "2",
                            Patterns::Integer(0),
                            "Global refinement level");

            prm.declare_entry("Grid scale", "1e-3",
                            Patterns::Double(0.0),
                            "Global grid scaling factor");

            prm.declare_entry("Pressure ratio p/p0", "100",
                            Patterns::Selection("20|40|60|80|100"),
                            "Ratio of applied pressure to reference pressure");
        }
        prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Geometry");
        {
            global_refinement = prm.get_integer("Global refinement");
            scale = prm.get_double("Grid scale");
            p_p0 = prm.get_double("Pressure ratio p/p0");
        }
        prm.leave_subsection();
    }

    // @sect4{Materials} =============================================================================
    void Materials::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Material properties");
        {
            prm.declare_entry("Poisson's ratio", "0.4999",
                            Patterns::Double(-1.0,0.5),
                            "Poisson's ratio");

            prm.declare_entry("Shear modulus", "80.194e6",
                            Patterns::Double(),
                            "Shear modulus");
        }
        prm.leave_subsection();
    }

    void Materials::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Material properties");
        {
            nu = prm.get_double("Poisson's ratio");
            mu = prm.get_double("Shear modulus");
        }
        prm.leave_subsection();
    }

    // @sect4{Linear solver} ================================================================================
    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Linear solver");
        {
            prm.declare_entry("Solver type", "CG",
                            Patterns::Selection("CG|Direct"),
                            "Type of solver used to solve the linear system");

            prm.declare_entry("Residual", "1e-6",
                            Patterns::Double(0.0),
                            "Linear solver residual (scaled by residual norm)");

            prm.declare_entry("Max iteration multiplier", "1",
                            Patterns::Double(0.0),
                            "Linear solver iterations (multiples of the system matrix size)");

            prm.declare_entry("Use static condensation", "true",
                            Patterns::Bool(),
                            "Solve the full block system or a reduced problem");

            prm.declare_entry("Preconditioner type", "ssor",
                            Patterns::Selection("jacobi|ssor"),
                            "Type of preconditioner");

            prm.declare_entry("Preconditioner relaxation", "0.65",
                            Patterns::Double(0.0),
                            "Preconditioner relaxation value");
        }
        prm.leave_subsection();
    }

    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Linear solver");
        {
            type_lin = prm.get("Solver type");
            tol_lin = prm.get_double("Residual");
            max_iterations_lin = prm.get_double("Max iteration multiplier");
            use_static_condensation = prm.get_bool("Use static condensation");
            preconditioner_type = prm.get("Preconditioner type");
            preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
        }
        prm.leave_subsection();
    }

    // @sect4{Nonlinear solver} ================================================================
    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Nonlinear solver");
        {
            prm.declare_entry("Max iterations Newton-Raphson", "10",
                            Patterns::Integer(0),
                            "Number of Newton-Raphson iterations allowed");

            prm.declare_entry("Tolerance force", "1.0e-9",
                            Patterns::Double(0.0),
                            "Force residual tolerance");

            prm.declare_entry("Tolerance displacement", "1.0e-6",
                            Patterns::Double(0.0),
                            "Displacement error tolerance");
        }
        prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Nonlinear solver");
        {
            max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
            tol_f = prm.get_double("Tolerance force");
            tol_u = prm.get_double("Tolerance displacement");
        }
        prm.leave_subsection();
    }

    // @sect4{Time} ===============================================================================
    void Time::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Time");
        {
            prm.declare_entry("End time", "1",
                            Patterns::Double(),
                            "End time");

            prm.declare_entry("Time step size", "0.1",
                            Patterns::Double(),
                            "Time step size");
        }
        prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("Time");
        {
            end_time = prm.get_double("End time");
            delta_t = prm.get_double("Time step size");
        }
        prm.leave_subsection();
    }

    // @sect4{All parameters} ======================================================================

    AllParameters::AllParameters(const std::string &input_file)
    {
        ParameterHandler prm;
        declare_parameters(prm);
        prm.parse_input(input_file);
        parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
        FESystem::declare_parameters(prm);
        Geometry::declare_parameters(prm);
        Materials::declare_parameters(prm);
        LinearSolver::declare_parameters(prm);
        NonlinearSolver::declare_parameters(prm);
        Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
        FESystem::parse_parameters(prm);
        Geometry::parse_parameters(prm);
        Materials::parse_parameters(prm);
        LinearSolver::parse_parameters(prm);
        NonlinearSolver::parse_parameters(prm);
        Time::parse_parameters(prm);
    }
}
