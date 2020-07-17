IMG = SST_diff.png SST_compare.png SSS_diff.png SSS_compare.png moc.png

all: $(IMG)

test_data:
	curl -o test_data.tar.gz ftp://ftp.gfdl.noaa.gov/pub/John.Krasting/om4labs/test_data.tar.gz
	tar -xzvf test_data.tar.gz
	rm -f test_data.tar.gz

moc.png: test_data ../om4labs/diags/moc/moc.py
	om4labs moc --model OM4p125 --platform testing test_data/output/ocean_month_z_d2_refined.0036-0040.ann.nc

SST_diff.png: test_data ../om4labs/diags/sst_annual_bias_1x1deg/sst_annual_bias_1x1deg.py
	om4labs sst_annual_bias_1x1deg -S diff --platform testing test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SSS_diff.png: test_data ../om4labs/diags/sss_annual_bias_1x1deg/sss_annual_bias_1x1deg.py
	om4labs sss_annual_bias_1x1deg -S diff --platform testing test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SST_compare.png: test_data ../om4labs/diags/sst_annual_bias_1x1deg/sst_annual_bias_1x1deg.py
	om4labs sst_annual_bias_1x1deg -S compare --platform testing test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SSS_compare.png: test_data ../om4labs/diags/sss_annual_bias_1x1deg/sss_annual_bias_1x1deg.py
	om4labs sss_annual_bias_1x1deg -S compare --platform testing test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

clean:
	rm -f $(IMG)

distclean:
	rm -f $(IMG)
	rm -fR test_data
