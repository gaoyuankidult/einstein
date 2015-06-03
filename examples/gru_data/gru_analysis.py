import einstein as E

with open("gru_record.txt") as f:
    all_content = f.read()
    all_content = all_content.replace(']][', ',')
    all_content = all_content.replace('[',' ')
    all_content = all_content.replace(']',' ')
    all_content = all_content.split(',')
    all_content = E.tools.make_chunks(all_content, 6)

    best_values = []
    for each in all_content:
        each_f = [float(e) for e in each]
        best_values.append(max(each_f))
    print(best_values)
    print "average of all values:", E.tools.mean(best_values)
    print "number of all values:", len(best_values)

