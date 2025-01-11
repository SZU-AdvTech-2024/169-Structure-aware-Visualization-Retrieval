for i in train test
    do
    mkdir ../full_VizML+/${i}_graph
    mkdir ../full_VizML+/${i}_json
    for j in line bar scatter box histogram 
        do
            echo $i $j
            mkdir ../full_VizML+/${i}_graph/${j}
            mkdir ../full_VizML+/${i}_json/${j}
            node parse.js $i $j
        done
    done