# Pe : 1.570796e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-10pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+02
# n time steps : 1000
# n output : 250
# n step/output : 4.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-10000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 1.000000e+01 -tend 6.250000e+00 -toutput 2.500000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 7.853982e+01
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-10pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.853982e+02
# n time steps : 1000
# n output : 125
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-5000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 5.000000e+00 -tend 6.250000e+00 -toutput 5.000000e-02 --time_step 6.250000e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 3.141593e+01
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-10pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+01
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.256637e+02
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+02
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+02
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 9.424778e+02
# n time steps : 1600
# n output : 100
# n step/output : 16.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-2000mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 2.000000e+00 -tend 6.250000e+00 -toutput 6.250000e-02 --time_step 3.906250e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.099557e+01
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-10pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.199115e+01
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.398230e+01
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.099557e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.199115e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.298672e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.398230e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-400pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 5.497787e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-500pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.597345e+02
# n time steps : 1120
# n output : 140
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-600pm-700mHz-D0 --resistance -1 -xi 100e-4 -ai 6.000000e-01 -fi 7.000000e-01 -tend 6.250000e+00 -toutput 4.464286e-02 --time_step 5.580357e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 3.141593e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-10pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-02 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.256637e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 9.424778e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.256637e+02
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-400pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+02
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-500pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.884956e+02
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-600pm-200mHz-D0 --resistance -1 -xi 100e-4 -ai 6.000000e-01 -fi 2.000000e-01 -tend 1.500000e+01 -toutput 7.812500e-02 --time_step 9.765625e-03 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.570796e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-20pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-02 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-40pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-02 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.853982e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.570796e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.356194e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-400pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.926991e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-500pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.712389e+01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-600pm-50mHz-D0 --resistance -1 -xi 100e-4 -ai 6.000000e-01 -fi 5.000000e-02 -tend 6.000000e+01 -toutput 3.125000e-01 --time_step 3.906250e-02 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 1.570796e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-100pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 1.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 3.141593e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 4.712389e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 6.283185e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-400pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 7.853982e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-500pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 9.424778e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-600pm-10mHz-D0 --resistance -1 -xi 100e-4 -ai 6.000000e-01 -fi 1.000000e-02 -tend 3.000000e+02 -toutput 1.562500e+00 --time_step 1.953125e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

# Pe : 9.424778e-01
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-200pm-3mHz-D0 --resistance -1 -xi 100e-4 -ai 2.000000e-01 -fi 3.000000e-03 -tend 1.000000e+03 -toutput 5.208333e+00 --time_step 6.510417e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.413717e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-300pm-3mHz-D0 --resistance -1 -xi 100e-4 -ai 3.000000e-01 -fi 3.000000e-03 -tend 1.000000e+03 -toutput 5.208333e+00 --time_step 6.510417e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 1.884956e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-400pm-3mHz-D0 --resistance -1 -xi 100e-4 -ai 4.000000e-01 -fi 3.000000e-03 -tend 1.000000e+03 -toutput 5.208333e+00 --time_step 6.510417e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.356194e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-500pm-3mHz-D0 --resistance -1 -xi 100e-4 -ai 5.000000e-01 -fi 3.000000e-03 -tend 1.000000e+03 -toutput 5.208333e+00 --time_step 6.510417e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

# Pe : 2.827433e+00
# n time steps : 1536
# n output : 192
# n step/output : 8.000000
python3 ../fbb_DD/PVSSAS_simulation.py -j global-600pm-3mHz-D0 --resistance -1 -xi 100e-4 -ai 6.000000e-01 -fi 3.000000e-03 -tend 1.000000e+03 -toutput 5.208333e+00 --time_step 6.510417e-01 --diffusion_coef 2.000000e-08 -lpvs 2.000000e-02 -nr 8 --sigma 1e-4 -rv 8.000000e-04 -rpvs 1.050000e-03 &

wait

#total number of jobs : 50