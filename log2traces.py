#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

def zf(s):
    s=str(s)
    if len(s)==1:
        return "0"+s;
    else:
        return s;

def log_to_gephi_traces(log,urldata,filename):

    # Initial checking
#    if 'id'!=nodes.columns[0]:
#        print("(log_to_gephi_traces) ERROR: node table's first column is not 'id'. Execution skipped.")
#        return;
#    if 'requested_id'!=edges.columns[1] or 'referrer_id'!=edges.columns[0]:
#        print("(log_to_gephi_traces) ERROR: edges table's first columns are not 'referrer_id' and 'requested_id'. Execution skipped.")
#        return;
#    if 'timestamp'!=edges.columns[2]:
#        print("(log_to_gephi_traces) ERROR: edges table's third column is not 'timestamp'. Execution skipped.")
#        return;
    # printing to gephi-readable file
    with open(filename, 'w') as f:
        f.write("graph\n[\n")
        node_counter=0
        for row in urldata.itertuples():
            f.write("  node\n  [\n")
            f.write("   id %s\n"%row.id)
            for col in range(1,urldata.shape[1]):
                f.write("   %s \"%s\"\n"%(urldata.columns[col],urldata.iloc[node_counter,col]))
            f.write("  ]\n")
            node_counter+=1
        for row in log.itertuples():
            f.write("  edge\n  [\n")
            f.write("   source %s\n"%urldata[urldata.url==row.referrer_url].iloc[0].id)
            f.write("   target %s\n"%urldata[urldata.url==row.requested_url].iloc[0].id)
            f.write("   timestamp \"%s\"\n"%(zf(pd.Timestamp(row.timestamp).hour)+':'+zf(pd.Timestamp(row.timestamp).minute)+':'+zf(pd.Timestamp(row.timestamp).second)))
            f.write("  ]\n")
        f.write("]\n")
    return;