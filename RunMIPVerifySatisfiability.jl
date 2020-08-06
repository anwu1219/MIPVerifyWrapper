# julia RunMIPVerifySatisfiability.jl --environment_path /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/ --property_file /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/acas_property_3.txt --network_file /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Networks/ACASXu/ACASXU_experimental_v2a_2_1.nnet --output_file /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/test_output.txt --tightening lp --timeout_per_node 20

# To run a simple test:
# module test
#        ARGS = ["--environment_path", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/", "--property_file", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/acas_property_3.txt", "--network_file", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Networks/ACASXu/ACASXU_experimental_v2a_2_1.nnet", "--output_file", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/test_output.txt", "--tightening", "lp", "--timeout_per_node", "20"]
#        include("RunMIPVerifySatisfiability.jl")
# end


using Pkg

# Interface:
# RunMIPVerifySatisfiability environment_path property.txt network.nnet output_file strategy timeout_per_node
# For parsing the arguments to the file
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper"
    "--property_file"
        help = "Property file name"
        arg_type = String
        required = true
    "--network_file"
        help = "Network file name"
        arg_type = String
        required = true
    "--output_file"
        help = "Output file name"
        arg_type = String
        required = true
    "--tightening"
        help = "Tightening strategy - mip, lp or ia"
        arg_type = String
        default = "mip"
    "--timeout_per_node"
        help = "Timeout in seconds per node"
        arg_type = Float64
        default = 20.0
    "--num_threads"
        help = "Number of threads"
        arg_type = Int64
        default = 1
    "--set_obj"
        help = "Set an objective for satisfiability check"
	arg_type = Bool
	default = false
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println(ARGS)
println(parsed_args)
environment_path = parsed_args["environment_path"]
property_file_name = parsed_args["property_file"]
network_file_name = parsed_args["network_file"]
tightening = parsed_args["tightening"]
timeout_per_node = parsed_args["timeout_per_node"]
output_file_name = parsed_args["output_file"]
num_threads = parsed_args["num_threads"]
set_obj = parsed_args["set_obj"]

Pkg.activate(environment_path)

using Interpolations
using NPZ
using JuMP
using ConditionalJuMP
using LinearAlgebra
using MathProgBase
using CPUTime

using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using MAT
using GLPKMathProgInterface
# You can use your solver of choice
using Gurobi



include(string(environment_path, "MIPVerify.jl/src/MIPVerify.jl"))
MIPVerify.setloglevel!("info")

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include(string(environment_path, "activation.jl"))
include(string(environment_path, "network.jl"))
include(string(environment_path, "util.jl"))


# Decide on your bound tightening strategy
if tightening == "lp"
   strategy = MIPVerify.lp
elseif tightening == "mip"
   strategy = MIPVerify.mip
elseif tightening == "ia"
    strategy = MIPVerify.interval_arithmetic
else
    println("Didn't recognize the tightening strategy")
    @assert false
end

# Read in the network and convert to a MIPVerify network
network = read_nnet(network_file_name)
mipverify_network = network_to_mipverify_network(network, "test", strategy)
num_inputs = size(network.layers[1].weights, 2)

# Run simple problem to avoid Sherlock startup time being counted
start_time = time()
println("Starting simple example")
simple_nnet = read_nnet(string(environment_path, "Networks/small_nnet.nnet"))
simple_mipverify_network = network_to_mipverify_network(simple_nnet)
simple_property_lines = readlines(string(environment_path, "Properties/small_nnet_property.txt"))
simple_lower_bounds, simple_upper_bounds = bounds_from_property_file(simple_property_lines, 1, simple_nnet.lower_bounds, simple_nnet.upper_bounds)

temp_p = get_optimization_problem(
      (1,),
      simple_mipverify_network,
      GurobiSolver(),
      lower_bounds=simple_lower_bounds,
      upper_bounds=simple_upper_bounds,
      )

add_output_constraints_from_property_file!(temp_p.model, temp_p.output_variable, simple_property_lines)
solve(temp_p.model)
println("Finished simple solve in: ", time() - start_time)

# Read in the input upper and lower bounds from the property file
property_lines = readlines(property_file_name)
lower_bounds, upper_bounds = bounds_from_property_file(property_lines, num_inputs, network.lower_bounds, network.upper_bounds)


# Propagate bounds using MIPVerify
# Start timing
CPUtic()

main_solver = GurobiSolver(Threads=num_threads, BestObjStop=(set_obj ? 0.5 : 0))
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node, Threads=num_threads)

p1 = get_optimization_problem(
      (num_inputs,),
      mipverify_network,
      main_solver,
      lower_bounds=lower_bounds,
      upper_bounds=upper_bounds,
      tightening_solver=tightening_solver,
      summary_file_name=string(output_file_name, ".bounds.txt")
      )

preprocessing_time = CPUtoc()

# Write to the output file the status, objective value, and elapsed time
preprocess_file = string(output_file_name, ".preprocess") # add on the .csv
open(preprocess_file, "w") do f
       	 write(f,
       	 string(preprocessing_time), "\n")
	 close(f)
end

println("Preprocessing took: ", preprocessing_time)

CPUtic()
# Add output constraints
add_output_constraints_from_property_file!(p1.model, p1.output_variable, property_lines)

# Add an objetive of 0 since we're just concerned with feasibility
# TODO: Find if there's a way to set it to be a feasibility problem that would be more efficient
if set_obj
   #@objective(p1.model, Min, max(abs(p1.input_variable .- centroid) ./ (upper_bounds .- lower_bounds)))
   centroid = ( lower_bounds + upper_bounds ) / 2
   @objective(p1.model, Min, MIPVerify.get_norm(Inf, (p1.input_variable - centroid) / (upper_bounds - lower_bounds)))
else
    @objective(p1.model, Max, 0)
 end


# Solve the feasibility problem
status = solve(p1.model)
solve_time = CPUtoc()

if (status == :Infeasible)
      println("Infeasible, UNSAT")
elseif (status == :InfeasibleOrUnbounded)
      println("Infeasible or unbounded, UNSAT")
elseif (status == :Optimal)
      println("Optimal, SAT")
elseif (status == :UserObjLimit)
      println("SAT")
else
      println("Unknown!")
end

# Does it keep any of the old objectives???????
# It still has an objective, where does that come from?

println("Preprocessing time: ", preprocessing_time)
println("Solve time: ", solve_time)
println("Percent preprocessing: ", round(100 * preprocessing_time / (preprocessing_time + solve_time), digits=2), "%")

# Write to the output file the status, objective value, and elapsed time
output_file = string(output_file_name) # add on the .csv
open(output_file, "w") do f
    # Writeout our results
    if ( status == :Infeasible || status == :InfeasibleOrUnbounded )
       	 write(f,
       	 "unsat ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    elseif ( status == :Optimal || status == :UserObjLimit )
       	 write(f,
       	 "sat ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    else
       	 write(f,
       	 "unknown ", 
       	 string(preprocessing_time + solve_time), " ",
       	  "\n")
    end
   close(f)
end
