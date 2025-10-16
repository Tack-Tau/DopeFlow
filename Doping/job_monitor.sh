#!/bin/bash
calc_type="$1"
calc_dir="$PWD"
rm nohup.out 2> /dev/null
if [ -f $calc_dir/job_${calc_type}.log ]
    then
    job_count="$(echo "$(tail -1 $calc_dir/job_${calc_type}.log | awk -F ' ' '{print $2}') + 1" | bc)"
else
    job_count=1
fi

if [ -f $calc_dir/diverge_structs ] ; then
    diverge_structs="$calc_dir/diverge_structs"
else
    touch diverge_structs
    diverge_structs="$calc_dir/diverge_structs"
fi

while [[ 1 ]]
    do
    N_dir=$( cat -n ../aflow_sym/uniq_poscar_list | tail -1 | awk -F ' ' '{print $1}' )
    while [[ $job_count -le $N_dir ]]
        do
        sq_results=$(squeue -u $USER | sed 1d | wc -l)
        if [ $sq_results -lt 60 ]
            then
            for i in $(seq 1 10)
                do
                current_struct="$(sed "${job_count}q;d" ../aflow_sym/uniq_poscar_list)"
                if [ $job_count -le $N_dir ] && ! grep -Fxq "$current_struct" "$diverge_structs"
                    then
                    cd $calc_dir/$job_count
                    if [ ! -e $calc_type ]
                        then
                        mkdir -p $calc_type
                    else
                        rm -rf $calc_type
                        mkdir -p $calc_type
                    fi
                    if [[ $calc_type == "Relax" ]]
                        then
                        cp ./POSCAR ./$calc_type/
                        cd $calc_type
                        cp $calc_dir/INCAR_$calc_type ./INCAR
                        cp $calc_dir/sbp_${calc_type}.sh ./sbp.sh
                        # cp $calc_dir/POTCAR ./POTCAR
                        sbatch sbp.sh
                        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                        job_count=$((job_count+1))
                    elif [[ $calc_type == "SC" ]]
                        then
                        cp ./Relax/CONTCAR ./$calc_type/POSCAR
                        cd $calc_type
                        cp $calc_dir/INCAR_$calc_type ./INCAR
                        cp $calc_dir/sbp_${calc_type}.sh ./sbp.sh
                        # cp $calc_dir/POTCAR ./POTCAR
                        sbatch sbp.sh
                        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                        job_count=$((job_count+1))
                    elif [[ $calc_type == "ELF" ]]
                        then
                        cp ./Relax/CONTCAR ./$calc_type/POSCAR
                        cp ./SC/CHGCAR ./$calc_type/
                        cd $calc_type
                        cp $calc_dir/INCAR_$calc_type ./INCAR
                        cp $calc_dir/sbp_${calc_type}.sh ./sbp.sh
                        # cp $calc_dir/POTCAR ./POTCAR
                        sbatch sbp.sh
                        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                        job_count=$((job_count+1))
                    elif [[ $calc_type == "Band" ]]
                        then
                        cp ./Relax/CONTCAR ./$calc_type/POSCAR
                        cp ./SC/CHGCAR ./$calc_type/
                        cd $calc_type
                        cp $calc_dir/INCAR_$calc_type ./INCAR
                        cp $calc_dir/sbp_${calc_type}.sh ./sbp.sh
                        # cp $calc_dir/POTCAR ./POTCAR
                        vaspkit -task 303 > vaspkit.out
                        mv POSCAR bk_POSCAR
                        cp PRIMCELL.vasp POSCAR
                        cp KPATH.in KPOINTS
                        sed -i "2 s/^   20/   40/" KPOINTS
                        sbatch sbp.sh
                        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                        job_count=$((job_count+1))
                    elif [[ $calc_type == "DOS" ]]
                        then
                        cp ./Relax/CONTCAR ./$calc_type/POSCAR
                        cp ./SC/CHGCAR ./$calc_type/
                        cd $calc_type
                        cp $calc_dir/INCAR_$calc_type ./INCAR
                        cp $calc_dir/sbp_${calc_type}.sh ./sbp.sh
                        # cp $calc_dir/POTCAR ./POTCAR
                        sbatch sbp.sh
                        echo "Job $job_count has been submitted" >> $calc_dir/job_${calc_type}.log
                        job_count=$((job_count+1))
                    else
                        echo "Wrong calculation type input, try again"
                    fi
                fi
            done
        else
            sleep 10
        fi
    done
done

echo "Job monitor ended"
