import numpy as np
import pickle as pkl

width = 15
mid_lookup_17_4_to_17_2 = []
mid_lookup_17_2_to_13_8 = []
mid_lookup_13_6_to_13_8 = []
modules_17_4 = 3192
modules_17_2 = 2520
modules_13_8 = 2142
modules_13_6 = 1638

def func_data(x, y, slope, offset):
    return abs(slope * x + offset - y)

def make_module_tables(vid, path_to_output):
    # currently only mid = 17 works
    if vid == 17:
        for i in range(modules_17_4):
            x = i+1
            temp = []
            for j in range(modules_17_2):
                y = j+1

                if (   ((func_data(x, y, 0.79, 239.4) <= width) and (x < 1500)) or 
                        (func_data(x, y, 0.79, 120.1) <= width) or
                        (func_data(x, y, 0.79, 3.1) <= width) or
                        ((func_data(x, y, 0.79, 363.0) <= width) and (x < 1500)) or
                        ((func_data(x, y, 0.79, -116.4 ) <= width) and (x > 1000)) or
                        ((func_data(x, y, 0.79, - 233.6 ) <= width)  and (x > 2000))or
                        ((func_data(x, y, 0.79, - 354.2 ) <= width) and (x > 2500))
                    ):
                    temp.append(y)
            mid_lookup_17_4_to_17_2.append(temp)
        #print(mid_lookup_17_4_to_17_2[211])


        for i in range(modules_17_2):
            x = i+1
            temp = []
            for j in range(modules_13_8):
                y = j+1
                if (   (func_data(x, y, 0.85, 0.4) <= width) or 
                        ((func_data(x, y, 0.85, 101.5) <= width) and (x < 1500))or
                        (func_data(x, y, 0.85, -100.2) <= width) or
                        ((func_data(x, y, 0.85, 204.1) <= width)  and (x < 1000))or
                        ((func_data(x, y, 0.85, -198.4 ) <= width) and (x > 1500))
                   ):
                    temp.append(y)
            mid_lookup_17_2_to_13_8.append(temp)

        # return mid_lookup_17_4_to_17_2, mid_lookup_17_2_to_13_8
        f = open('%s/mid_lookup_17_4_to_17_2.pkl' % path_to_output, 'wb')
        pkl.dump(mid_lookup_17_4_to_17_2, f)
        f = open('%s/mid_lookup_17_2_to_13_8.pkl' % path_to_output, 'wb')
        pkl.dump(mid_lookup_17_2_to_13_8, f)

    elif vid == 13:
        for i in range(modules_13_6):
            x = i+1
            temp = []
            for j in range(modules_13_8):
                y = j+1
                if (   (func_data(x, y, 1.31, -0.34) <= width) or 
                        ((func_data(x, y, 1.31, 100.3) <= width) and (x > 400) and (x < 1400))or
                        ((func_data(x, y, 1.31, 199.4) <= width) and (x > 700) and (x < 1300)) or
                        ((func_data(x, y, 1.31, -101.3) <= width)  and (x < 1200))or
                        ((func_data(x, y, 1.31, -204.6 ) <= width) and (x > 300) and (x < 800))
                   ):
                    temp.append(y)
            mid_lookup_13_6_to_13_8.append(temp)

        # return mid_lookup_17_4_to_17_2, mid_lookup_17_2_to_13_8
        f = open('%s/mid_lookup_13_6_to_13_8.pkl' % path_to_output, 'wb')
        pkl.dump(mid_lookup_13_6_to_13_8, f)
    else:
        print('warning!! choose volume id from (17, 13). Othre modules will be implemented.')
        # return 0, 0
