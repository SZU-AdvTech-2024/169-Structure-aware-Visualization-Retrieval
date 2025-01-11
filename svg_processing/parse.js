const fs = require('fs');
const svgp = require('svg-parser');
const cssj = require('cssjson');
const getBounds = require('svg-path-bounds');
var path = require('path');
var d3 = require('d3-regression')

const {SVGPathData} = require('svg-pathdata');

const set = process.argv[2]
const type = process.argv[3]

const max_char_num = 500
const max_vertex_num = 500

const input_path = `../full_VizML+/${set}_svg/${type}`;
const output_path = `../full_VizML+/${set}_json/${type}`;

var id = 1;

const disposed_class = ["bg", "legendtoggle", "scrollbar", "gridlayer", "zerolinelayer", "xtick", "ytick", "legend", "infolayer"]

const no_stroke_color = {
    stroke_red: 0,
    stroke_green: 0,
    stroke_blue: 0
}

const no_fill_color = {
    fill_red: 0,
    fill_green: 0,
    fill_blue: 0
}

var list_unable_resolve = []
var list_with_null = []
var num_exceed_max_char = 0

function get_line_area(data, size){
    var area = 0
    for (var i=1; i<data.length; i++){
        area += Math.sqrt((((data[i]['x']-data[i-1]['x'])/size[0])**2) + (((data[i]['y']-data[i-1]['y'])/size[1])**2))
    }
    return area
}

function check_and_get_property(parsed_data, attr){
    // check if the attr is in properties. If the value is nan or the attr does not exist, return 0
    return attr in parsed_data['properties'] ? (parseFloat(parsed_data['properties'][attr]) > 0 ?
            parseFloat(parsed_data['properties'][attr]) : 0) : 0
}

// https://www.geeksforgeeks.org/how-to-convert-rgb-color-string-into-an-object-in-javascript/
function rgbstring2obj(rgb, prefix){
    if (!rgb.includes("rgb") && prefix === 'fill'){return no_fill_color}
    if (!rgb.includes("rgb") && prefix === 'stroke'){return no_stroke_color}

    let colors = ["red", "green", "blue"]

    let colorArr = rgb.slice(
        rgb.indexOf("(") + 1, 
        rgb.indexOf(")")
    ).split(", ");
      
    let obj = new Object();

    colorArr.forEach((k, i) => {
        obj[prefix+"_"+colors[i]] = k/255
    })

    return obj
}

// Get features for <text>, translate = (x, y), size = (width, length)
function get_text_feature(parsed_data, translate, size){

    var len = parsed_data['properties']['data-unformatted'] ? parsed_data['properties']['data-unformatted'].toString().length : 0
    if (len > max_char_num) num_exceed_max_char += 1

    var x = check_and_get_property(parsed_data, 'x')
    var y = check_and_get_property(parsed_data, 'y')

    return {
        cx: (x+translate[0]) / size[0],
        cy: (y+translate[1]) / size[1],
        area: 0,
        dx: 0,
        dy: 0,
        length: len > max_char_num ? 1 : len/max_char_num,
        num_vertex: 0,
    }
}

function get_rect_feature(parsed_data, translate, size){
    var x = check_and_get_property(parsed_data, 'x')
    var y = check_and_get_property(parsed_data, 'y')

    var w = check_and_get_property(parsed_data, 'width')
    var h = check_and_get_property(parsed_data, 'height')

    return {
        cx: (x+translate[0]) / size[0],
        cy: (y+translate[1]) / size[1],
        area: (w * h) / (size[0] * size[1]),
        dx: (w / 2) / size[0],
        dy: (h / 2) / size[0],
        length: 0,
        num_vertex: 0,
    }
}

// Get features for <path>
function get_path_feature(parsed_data, translate, size){

        let [left, top, right, bottom] = getBounds(parsed_data['properties']['d'])
        if (parsed_data['properties']['class']){
            if ((!parsed_data['properties']['d'].includes('Z')) && (!["js-fill", "openline", "point", "mean"].includes(parsed_data['properties']['class'])) && (!parsed_data['properties']['class'].includes("crisp"))  && (!parsed_data['properties']['class'].includes("error"))){
                var pathData = new SVGPathData (parsed_data['properties']['d']);
    
                var data = pathData.commands.filter(d=>"x"in d && "y" in d).map(d => {
                    return {
                    x: d.x/size[0],
                    y: d.y/size[1]
                    }
                    })
    
                var data_length = data.length
    
                var regression = d3.regressionLoess()
                    .x(d => d.x)
                    .y(d => d.y)
                    .bandwidth(0.5);
                
                var feature_trend = regression(data)
    
                if  (feature_trend.length > 5){
    
                    let list_keep = [Math.round(feature_trend.length*0.0), Math.round((feature_trend.length-1)*0.25),
                            Math.round((feature_trend.length-1)*0.5), Math.round((feature_trend.length-1)*0.75), Math.floor((feature_trend.length-1)*1)]
    
    
                    var feature_trend_f =  feature_trend.filter((d, i)=> {
                            return list_keep.includes(i)
                        }).map(d=>d[1])
                    if (feature_trend_f.length !==5) console.log("if", list_keep, feature_trend_f.length)
                }
                else{
                    var feature_trend_f = feature_trend.map(d=>d[1]).concat(Array(5-feature_trend.length).fill(0))
                    if (feature_trend_f.length !==5) console.log("else", feature_trend_f.length, Array(5-data_length).fill(0))
                }
    
                
                return {
                    // local position + global translation
                    cx: ((right + left) / 2 + translate[0]) / size[0],
                    cy: ((top + bottom) / 2 + translate[1]) / size[1], 
                    // normalized cover area of the path
                    area:  get_line_area(data, size),
                    // normalized dx
                    dx: 0,
                    // normalized dy
                    dy: 0,
                    length: 0,
                    num_vertex: parsed_data['properties']['d'].match(/[a-zA-Z]/g).length > max_vertex_num ? 1: 
                        parsed_data['properties']['d'].match(/[a-zA-Z]/g).length/max_vertex_num ,
                    feature_trend_0: feature_trend_f[0],
                    feature_trend_1: feature_trend_f[1],
                    feature_trend_2: feature_trend_f[2],
                    feature_trend_3: feature_trend_f[3],
                    feature_trend_4: feature_trend_f[4],
                }
                
    
            }
        }
        

        return {
            // local position + global translation
            cx: ((right + left) / 2 + translate[0]) / size[0],
            cy: ((top + bottom) / 2 + translate[1]) / size[1], 
            // normalized cover area of the path
            area: ((right - left) * (bottom - top)) / (size[0] * size[1]),
            // normalized dx
            dx: ((right - left) / 2) / size[0],
            // normalized dy
            dy: ((bottom - top) / 2) / size[1],
            length: 0,
            num_vertex: parsed_data['properties']['d'].match(/[a-zA-Z]/g).length > max_vertex_num ? 1: 
                parsed_data['properties']['d'].match(/[a-zA-Z]/g).length/max_vertex_num ,
        }
        console.log(parsed_data['properties'])
        return {
            cx: translate[0] / size[0],
            cy: translate[1] / size[1], 
            // normalized cover area of the path
            area: 0,
            // normalized dx
            dx: 0,
            // normalized dy
            dy: 0,
            length: 0,
            num_vertex: 0,
            ...type
        }
}

const get_feature = {
    text: get_text_feature,
    path: get_path_feature,
    rect: get_rect_feature
}

function get_non_empty_g(parsed_data, filename, translate=[0, 0], size=[0, 0]){
    var list_non_empty = []

    // keep elements without children
    if (!('children' in parsed_data)){
        return 0
    }

    // remove gridlayer and defs
    if (disposed_class.includes(parsed_data['properties']['class']) ||  parsed_data['tagName'] =='defs'){
        return 0
    }

    // remove useless path or undefined path
    if ((parsed_data['tagName'] == 'path' && parsed_data['properties']['d'] == "M0,0") || (parsed_data['tagName'] == 'path' && !('d' in parsed_data['properties'])) || (parsed_data['tagName'] == 'path' && parsed_data['properties']['d'].length < 1) ){
        return 0
    }

    if (parsed_data['children'].length === 0 && parsed_data['tagName'] === 'g'){
        return 0 
    }

    if (parsed_data['children'].length === 0 && parsed_data['tagName'] === 'svg'){
        return 0 
    }

    else {

        // if translate, plus
        if ('transform' in parsed_data['properties']){

            if (parsed_data['properties']['transform'].includes("translate")){
                var translate_string = parsed_data['properties']['transform'].split(")")[0].split("(")[1]
                
                translate = [parseFloat(translate_string.split(",")[0]) + translate[0], 
                parseFloat(translate_string.split(",")[1]) + translate[1]]

            }
            else{
                // if no translate exists in the transform, and the element is not text, and transform is valid, log for debug
                if (!(parsed_data['tagName'] === 'text') && !(list_unable_resolve.includes(filename)) 
                        && parsed_data['properties']['transform'].length > 0 ){

                    // if annotation in class name, do not consider it as an error
                    if ('class' in parsed_data['properties']){
                        if (!(parsed_data['properties']['class'].includes('annotation'))) list_unable_resolve.push(filename)
                    }
                    else list_unable_resolve.push(filename)
                    
                }
            }
        }

        if ('width' in parsed_data['properties'] && parsed_data['tagName'] === 'svg'){
            size = [parseFloat(parsed_data['properties']['width']), parseFloat(parsed_data['properties']['height'])]
        }

        for (var i in parsed_data['children']){
            var return_value = get_non_empty_g(parsed_data['children'][i], filename, translate, size)
            if (return_value !== 0){
                list_non_empty.push(return_value)
            }
        }
    }

    if (list_non_empty.length > 0 && parsed_data['tagName'] === 'g'){
        var type = parsed_data['tagName'] == undefined ? "un" : parsed_data['tagName']
        var class_name = parsed_data['properties']['class'] == undefined ? "un" : parsed_data['properties']['class']
        parsed_data['id'] = `${type}_${class_name}_${id++}`

        // different lines is actually cross-group. thus delta is always 0.
        if (parsed_data['properties']['class'] == "points" || parsed_data['properties']['class'] == "lines"){
            if (list_non_empty[0]['children'].length == 0){
                list_non_empty.sort((a, b) => {
                    if (a['features']['cx'] !== b['features']['cx']){
                        return a['features']['cx'] - b['features']['cx']
                    }
                    else{
                        return a['features']['cy'] - b['features']['cy']
                    }
                    
                })

                for (var i in list_non_empty){
                    if (i == 0){
                        list_non_empty[i]['features']['delta_x'] = 0
                        list_non_empty[i]['features']['delta_y'] = 0
                    }
                    else{
                        list_non_empty[i]['features']['pre_node'] = list_non_empty[i-1]['id']
                        list_non_empty[i]['features']['delta_x'] = list_non_empty[i]['features']['cx'] - list_non_empty[i-1]['features']['cx']
                        list_non_empty[i]['features']['delta_y'] = list_non_empty[i]['features']['cy'] - list_non_empty[i-1]['features']['cy']
                    }
                }
            } 
            else{
                list_non_empty.sort((a, b) => {
                    if (a['children'][0]['features']['cx'] !== b['children'][0]['features']['cx']){
                        return a['children'][0]['features']['cx'] - b['children'][0]['features']['cx']
                    }
                    else {
                        return a['children'][0]['features']['cy'] - b['children'][0]['features']['cy']
                    }
                })
               
                
                for (var i in list_non_empty){
                    list_non_empty[i]['children'][0]['children']
                    if (i == 0){
                        list_non_empty[i]['children'][0]['features']['delta_x'] = 0
                        list_non_empty[i]['children'][0]['features']['delta_y'] = 0
                    }
                    else{
                        list_non_empty[i]['children'][0]['features']['pre_node'] = list_non_empty[i-1]['children'][0]['id']
                        list_non_empty[i]['children'][0]['features']['delta_x'] = list_non_empty[i]['children'][0]['features']['cx'] - list_non_empty[i-1]['children'][0]['features']['cx']
                        list_non_empty[i]['children'][0]['features']['delta_y'] = list_non_empty[i]['children'][0]['features']['cy'] - list_non_empty[i-1]['children'][0]['features']['cy']
                    }
                }
            }

            
        }

        

        parsed_data['children'] = list_non_empty

        return parsed_data
    }
    else if (parsed_data['tagName'] === 'svg'){
        parsed_data['children'] = list_non_empty

        // Type: <g>, <path>, ...
        var type = parsed_data['tagName'] == undefined ? "un" : parsed_data['tagName']
        // Class: infolayer, xtick, ...
        var class_name = parsed_data['properties']['class'] == undefined ? "un" : parsed_data['properties']['class']
        parsed_data['id'] = `${type}_${class_name}_${id++}`

        return parsed_data
    }
    else if (parsed_data['tagName'] !== 'g'){
        parsed_data['children'] = list_non_empty

        // Type: <g>, <path>, ...
        var type = parsed_data['tagName'] == undefined ? "un" : parsed_data['tagName']
        // Class: infolayer, xtick, ...
        var class_name = parsed_data['properties']['class'] == undefined ? "un" : parsed_data['properties']['class']
        parsed_data['id'] = `${type}_${class_name}_${id++}`
        
        var css_json= cssj.toJSON(parsed_data['properties']['style'])
        var stroke_color = "stroke" in css_json['attributes'] && css_json['attributes']['stroke'] !== 'none'? 
            rgbstring2obj(css_json['attributes']["stroke"], "stroke") : no_stroke_color

        var fill_color = "fill" in css_json['attributes'] && css_json['attributes']['fill'] !== 'none' ? 
            rgbstring2obj(css_json['attributes']["fill"], "fill") : no_fill_color

        if (parsed_data['tagName'] in get_feature){
            var element_feature =  get_feature[parsed_data['tagName']](parsed_data, translate, size)
        }
        else{
            // if elements are not visual elements, just give them the same info as the layer above
            var element_feature = {
                cx: translate[0] / size[0],
                cy: translate[1] / size[1],
                area: 0,
                dx:0,
                dy:0,
                length: 0,
                num_vertex: 0,
            }
        }
        
        if (Array.isArray(element_feature)){
            element_feature.sort((a, b) => a['cx'] - b['cx'])
            parsed_data['features'] = {
                "class": class_name,
                "type": type,
            }
            parsed_data['children'] = []

            for (var i in element_feature){
                if (i == 0){
                    element_feature[i]['delta_x'] = 0
                    element_feature[i]['delta_y'] = 0
                }
                else{
                    element_feature[i]["pre_node"] = parsed_data['children'][i-1]['id']
                    element_feature[i]['delta_x'] = element_feature[i]['cx'] - element_feature[i-1]['cx']
                    element_feature[i]['delta_y'] = element_feature[i]['cy'] - element_feature[i-1]['cy']
                }
                parsed_data['children'].push(
                    {
                        id: `point_${class_name}_${id++}`,
                        features: {
                            // normalized
                            ...stroke_color,
                            // normalized
                            ...fill_color,
                            // (0, 1)
                            "fill_opacity": "fill-opacity" in css_json['attributes'] ? parseFloat(css_json['attributes']["fill-opacity"]) : 0, 
                            // (0, 1)
                            "stroke_opacity": "stroke-opacity" in css_json['attributes'] ? parseFloat(css_json['attributes']["stroke-opacity"]) : 0,
                            // normalized
                            "stroke_width": "stroke-width" in css_json['attributes'] ? (size[1] > size[0] ? parseFloat(css_json['attributes']["stroke-width"].split("px")[0])/size[1] : parseFloat(css_json['attributes']["stroke-width"].split("px")[0])/size[0]) : 0,
                            "data": parsed_data['properties']['data-unformatted'],
                            "class": class_name,
                            "type": 'point',
                            // normalized
                            ...element_feature[i]
                        }
                    }
                )
            }
        } 
        else{
            parsed_data['features'] = {
                // normalized
                ...stroke_color,
                // normalized
                ...fill_color,
                // (0, 1)
                "fill_opacity": "fill-opacity" in css_json['attributes'] ? parseFloat(css_json['attributes']["fill-opacity"]) : 0, 
                // (0, 1)
                "stroke_opacity": "stroke-opacity" in css_json['attributes'] ? parseFloat(css_json['attributes']["stroke-opacity"]) : 0,
                // normalized
                "stroke_width": "stroke-width" in css_json['attributes'] ? (size[1] > size[0] ? parseFloat(css_json['attributes']["stroke-width"].split("px")[0])/size[1] : parseFloat(css_json['attributes']["stroke-width"].split("px")[0])/size[0]) : 0,
                "data": parsed_data['properties']['data-unformatted'],
                "class": class_name,
                "type": type,
                // normalized
                ...element_feature
            }
        }
        

        delete parsed_data['properties']
        delete parsed_data['tagName']
        delete parsed_data['type']

        return parsed_data
    }
    else{
        return 0
    }
    
    
}

function get_parsed(data_processed, filename){
    var parsed_data = svgp.parse(data_processed)['children'][0]
    
    var write_data = get_non_empty_g(parsed_data, filename)

    return write_data
}

fs.readdir(input_path, function (err, files) {
    if (err) {
      console.error("Could not list the directory.", err);
      process.exit(1);
    }
    
    var count = 0
  
    files.forEach(function (file, index) {
      // Make one pass and make the file complete
        var fromPath = path.join(input_path, file);

        if (file.includes(".svg")){
            fs.readFile(fromPath, 'utf8' , (err, data) => {
                if (err) {
                    console.error(file, err)
                    return
                }
                
                // remove interaction layers
                try{
                    var data_processed_1 = data.split("<defs")[0]+'<g class="cartesianlayer">'+data.split("</svg>")[0].split('<g class="cartesianlayer">')[1].split('<g class="polarlayer">')[0]+"</svg>"
                    var data_processed_2 = data.split("</svg>")[1].split("</div>")[1]+"</svg>"
                }
                catch (error){
                    console.log(file)
                    console.error(error)
                }
                

                try{   
                    var data_parsed_1 = get_parsed(data_processed_1, file)
                    
                    var write_data = data_parsed_1

                }
                catch (error){
                    console.log(file)
                    console.error(error)
                }
                
                id = 0

                fs.writeFile(`debug_${type}.json`, JSON.stringify({"transform_unresovable": list_unable_resolve, 
                                                                    "null_feature": list_with_null, 
                                                                    "exceed_max_char": num_exceed_max_char}), function (err,data) {
                    if (err) {
                    return console.log(err);
                    }
                });
                
                try{
                    fs.writeFile(`${output_path}/${file.slice(0, -4)}.json`, JSON.stringify(write_data), function (err,data) {
                        if (err) {
                        return console.log(err);
                        }
                        count += 1
                        if (count%100 == 0) console.log(count);
                    }); 
                }
                catch (error){
                    fs.writeFile(`${output_path}/${file.slice(0, -4)}.json`, JSON.stringify({}), function (err,data) {
                        if (err) {
                        return console.log(err);
                        }
                        count += 1
                        if (count%100 == 0) console.log(count);
                    }); 
                    }               
            })
        }
    });
  });
