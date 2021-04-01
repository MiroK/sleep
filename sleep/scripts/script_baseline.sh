# Pe : 1.570796e+01
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+01
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.853982e+01
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.712389e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.413717e+01
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.827433e+01
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.068583e+01
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.413717e+02
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.827433e+02
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.241150e+02
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 5.654867e+02
# n time steps : 1800
# n output : 225
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-9000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 9.000000e+00 -tend 6.250000e+00 -toutput 2.777778e-02 --time_step 3.472222e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.256637e+01
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.513274e+01
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+01
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.256637e+02
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.513274e+02
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.769911e+02
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 5.026548e+02
# n time steps : 1600
# n output : 200
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-8000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 8.000000e+00 -tend 6.250000e+00 -toutput 3.125000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.099557e+01
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.199115e+01
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 5.497787e+01
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.099557e+02
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.199115e+02
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.298672e+02
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.398230e+02
# n time steps : 1400
# n output : 175
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-7000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 7.000000e+00 -tend 6.250000e+00 -toutput 3.571429e-02 --time_step 4.464286e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 9.424778e+00
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.884956e+01
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.712389e+01
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 9.424778e+01
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.884956e+02
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.827433e+02
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.769911e+02
# n time steps : 1200
# n output : 150
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-6000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 6.000000e+00 -tend 6.250000e+00 -toutput 4.166667e-02 --time_step 5.208333e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 7.853982e+00
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-1pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-03 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+01
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-2pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-03 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.926991e+01
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-5pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-03 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.853982e+01
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-10pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-20pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.356194e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-30pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j baseline-40pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

#total number of jobs : 42