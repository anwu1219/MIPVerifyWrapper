# julia RunMIPVerifySatisfiability.jl --environment_path /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/ --base_path /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/ --property_file Properties/acas_property_3.txt --network_file Networks/ACASXu/ACASXU_experimental_v2a_2_1.nnet --output_file test_output.txt --tightening lp --timeout_per_node 20

# To run a simple test:
# module test
#        ARGS = ["--environment_path", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/", "--base_path", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/", "--property_file", "Properties/acas_property_3.txt", "--network_file", "Networks/ACASXu/ACASXU_experimental_v2a_2_1.nnet", "--output_file", "test_output.txt", "--tightening", "lp", "--timeout_per_node", "20"]
#        include("RunMIPVerifySatisfiability.jl")
# end


using Pkg
Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper")

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


# Interface:
# RunMIPVerifySatisfiability environment_path base_path property.txt network.nnet output_file strategy timeout_per_node
# For parsing the arguments to the file
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper"
    "--base_path"
        help = "Base path to the network and property files"
        arg_type = String
        default = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper"
    "--property_file"
        help = "Property file name relative to base_path"
        arg_type = String
        required = true
    "--network_file"
        help = "Network file name relative to base_path"
        arg_type = String
        required = true
    "--output_file"
        help = "Output file name relative to base_path"
        arg_type = String
        required = true
    "--tightening"
        help = "Tightening strategy - mip, lp or interval_analysis"
        arg_type = String
        default = "mip"
    "--timeout_per_node"
        help = "Timeout in seconds per node"
        arg_type = Float64
        default = 20.0
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println(ARGS)
println(parsed_args)
environment_path = parsed_args["environment_path"]
base_path = parsed_args["base_path"]
property_file_name = string(base_path, parsed_args["property_file"])
network_file_name = string(base_path, parsed_args["network_file"])
tightening = parsed_args["tightening"]
timeout_per_node = parsed_args["timeout_per_node"]
output_file_name = string(base_path, parsed_args["output_file"])

include(string(environment_path, "MIPVerify.jl/src/MIPVerify.jl"))
MIPVerify.setloglevel!("info") # "info", "notice"

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
elseif tightening == "interval_arithmetic"
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
simple_nnet = read_nnet(string(base_path, "Networks/small_nnet.nnet"))
simple_mipverify_network = network_to_mipverify_network(simple_nnet)
simple_property_lines = readlines(string(base_path, "Properties/small_nnet_property.txt"))
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

main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)


# Propagate bounds using MIPVerify
# Start timing
preprocessing_time = @CPUelapsed p1 = get_optimization_problem(
      (num_inputs,),
      mipverify_network,
      main_solver,
      lower_bounds=lower_bounds,
      upper_bounds=upper_bounds,
      tightening_solver=tightening_solver,
      summary_file_name="",
      )


println("Preprocessing took: ", preprocessing_time)

# See if any of the outputs can be chosen
statuses = String[]
solve_times = zeros(Float64, length(p1.output_variable))
# Set it to be a feasibility problem
@objective(p1.model, Max, 0)
num_outputs = length(p1.output_variable)

temp_p1 = deepcopy(p1) # Run once because it takes longer the first run of the function

# Iterate through each to see if it can be chosen
for i = 1:num_outputs
    cur_solve_time = @CPUelapsed begin
        temp_p1 = deepcopy(p1)

        # Add a constraint that the current output_var must be larger than all others
        for j = 1:num_outputs
            if (i != j)
                @constraint(temp_p1.model, temp_p1.output_variable[i] >= temp_p1.output_variable[j])
            end
        end

        # Just perform the feasibility problem
        status = solve(temp_p1.model)
        if (status == :Infeasible)
            push!(statuses, "unsat")
        elseif (status == :Optimal)
            push!(statuses, "sat")
        else
            push!(statuses, "timeout")
        end
    end
    solve_times[i] = cur_solve_time
end

# Add a constraint that the current output_var must be larger than all others
for j = 1:num_outputs
    if (5 != j)
        @constraint(p1.model, p1.output_variable[5] >= p1.output_variable[j])
    end
end
# See how large of a perturbation it can handle
temp_var_1 = @variable(p1.model)

# Starts at lower left corner of the box, see how large it can go in each direction
@constraint(p1.model, p1.input_variable[1] == lower_bounds[1])
@constraint(p1.model, p1.input_variable[2] == lower_bounds[2])
@constraint(p1.model, p1.input_variable[3] == lower_bounds[3])
@constraint(p1.model, p1.input_variable[4]- lower_bounds[4] <= temp_var_1)

# for (i, input) in enumerate(p1.input_variable)
#     center_input = (lower_bounds[i] + upper_bounds[i])/2
#     @constraint(p1.model, -radius_var <= p1.input_variable[i] - center_input)
#     @constraint(p1.model, p1.input_variable[i] - center_input <= radius_var)
# end
@objective(p1.model, Min, temp_var_1)
@CPUtime solve(p1.model)
println("Objective value: ", getobjectivevalue(p1.model))
println(getvalue(temp_var_1))




println(statuses)
solve_time = sum(solve_times)

println("Preprocessing time: ", preprocessing_time)
println("Solve times: ", solve_times)
println("Total solve times: ", solve_time)
println("Total time: ", preprocessing_time + solve_time)
println("Percent preprocessing: ", round(100 * preprocessing_time / (preprocessing_time + solve_time), digits=2), "%")