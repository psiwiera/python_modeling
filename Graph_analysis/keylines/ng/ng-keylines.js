//
//     Copyright © 2011-2015 Cambridge Intelligence Limited. 
//     All rights reserved.
//

// shim Object.create
if (typeof Object.create != 'function') {
  Object.create = (function() {
    var Temp = function() {};
    return function (prototype) {
      if (arguments.length > 1) {
        throw Error('Second argument not supported');
      }
      if (typeof prototype != 'object') {
        throw TypeError('Argument must be an object');
      }
      Temp.prototype = prototype;
      var result = new Temp();
      Temp.prototype = null;
      return result;
    };
  })();
}

// create new angular module
angular.module('keylines', []);

angular.module('keylines').
  factory('klComponent', ['$rootScope', function ($rootScope) {

    //Note this is complex due to some calls being async. Async calls change scope only once complete.
    function wrap(fn, afterFn) {
      return function() {
        var args = [].slice.call(arguments);
        //not a particularly safe test as the function might be called with the wrong args. would be better
        //long term to know which functions are async beforehand via a function definition map.
        var async = args.length && angular.isFunction(args[args.length-1]);
        if (async) {
          //invoke the function call but use a different callback
          var cb = args.pop();
          args.push(function() {
            var cbargs = [].slice.call(arguments);
            cb.apply(null, cbargs);
            afterFn();
          });
          return fn.apply(null, args);
        }
        else {
          //invoke the function
          var res = fn.apply(null, args);
          afterFn();
          //now we need to do some further wrapping in case the function has returned
          //an object that has further functions on it. this is the case for the graph and
          //combo namespaces
          return wrapObject(res, afterFn);
        }
      };
    }

    // Wrap the KeyLines object so that everytime a function is called we can do something (update our isolate scope)
    function wrapObject(obj, afterFn) {
      if (angular.isObject(obj)) {
        // wrap in the right way the given object
        var ret = angular.isArray(obj) ? [] : {};
        // use the angular iterator
        angular.forEach(obj, function (value, key){
          if (angular.isFunction(value)) {
            ret[key] = wrap(value, afterFn);
          } else {
            //just add the other values. note this means that the wrapping will
            //only work on functions on the top level object, not any sub object functions
            ret[key] = value;
          }
        });
        return ret;
      } else {
        //string, number, boolean, undefined, null
        return obj;
      }
    }

    var eventPrefix = 'kl',
      eventJoiner = ':';

    function ngName(name) {
      return 'kl' + capitalize(name);
    }

    function capitalize(name) {
      return name.charAt(0).toUpperCase() + name.slice(1);
    }

    // 3 events are broadcast, this is not inefficient as it only does something when there are listeners present
    function broadcast (eventName, componentId, eventData) {
      eventData = angular.isArray(eventData) ? eventData : [eventData];
      // broadcast an event based on our top-level event prefix 'kl'
      $rootScope.$broadcast.apply($rootScope, [eventPrefix, eventName, componentId].concat(eventData));
      // broadcast another event based on type (chart/timebar)
      $rootScope.$broadcast.apply($rootScope, [[eventPrefix, eventName].join(eventJoiner)].concat([componentId]).concat(eventData));
      // broadcast one more event scoped to kl prefix, type, and the id of the component
      $rootScope.$broadcast.apply($rootScope, [[eventPrefix, eventName, componentId].join(eventJoiner)].concat(eventData));
    }

    // component setup: things shared between timebar and chart
    var klComponent = {
      // helper method to configure keylines from directive attributes
      config: function (base) {
        KeyLines.paths({ 
          assets: base + 'assets/',
          flash: { 
            swf: base + 'swf/keylines.swf', 
            swfObject: base + 'js/swfobject.js', 
            expressInstall: base + 'swf/expressInstall.swf' 
          },
          images: base,
        });
      },
      wrapObject: wrapObject,
      broadcast: broadcast,
      // create a KeyLines component
      create: function (id, type, callback) {
        KeyLines.create({id: id, type: type}, function (err, component) {
          callback(component);
          // bind all KeyLines events to be broadcast by our angular event broadcast system
          component.bind('all', function (eventName) {
            if (eventName !== 'redraw') {
              var params = [type].concat(Array.prototype.slice.call(arguments, 1));
              broadcast(eventName, id, params);
            }
          });
        });
      }
    };
    return klComponent;
  }]);

angular.module('keylines').
  factory('klChartService', ['klComponent', function (klComponent) { 

    var klChartService = Object.create(klComponent);

    klChartService.create = function (elemId, callback) {
      klComponent.create(elemId, 'chart', callback);
    };

    return klChartService;

  }]);

angular.module('keylines').
  factory('klTimebarService', ['klComponent', function (klComponent) { 

    var klTimebarService = Object.create(klComponent);

    klTimebarService.create = function (elemId, callback) {
      klComponent.create(elemId, 'timebar', callback);
    };

    return klTimebarService;

  }]);

angular.module('keylines').
directive('klChart', ['klChartService', '$timeout', function (klChartService, $timeout) { 

  return {
    restrict: 'A',  

    scope: {
      klOptions: '=',
      klSelected: '=',
      klChart: '='
    },

    controller: function ($scope) {
      // changes to a user-defined klOptions object will update KeyLines chart accordingly
      $scope.$watch('klOptions', function (val) {
        if ($scope.chart && val) {
          $scope.chart.options(val);
        }
      }, true); //true to watch deep
      // changes to a user-defined klSelected will set chart properties
      $scope.$watch('klSelected', function (val) {
        if ($scope.chart && val) {
          $scope.chart.setProperties(val);
        }
      }, true); //true to watch deep
    },

    link: function (scope, element, attrs) {
      var id = element[0].id;

      scope.$on('kl:selectionchange:' + id, function (event) {
        scope.$apply(function () {
          scope.klSelected = scope.chart.getItem(scope.chart.selection());
        });
      });

      function updateIsolateScope () {
        // needs timeout so that the $watch in controller can run first... (try not using $timeout then clicking turnGrey in angular demo and only tb goes grey)
        $timeout(function () {
          scope.klSelected = scope.chart.getItem(scope.chart.selection());
          scope.klOptions = scope.chart.options();
        });
      }

      //the base path is where KeyLines will load both its core files (the swf files, etc) and also its assets
      // - must include an ending / slash
      klChartService.config(attrs.klBasePath);

      // finally, we will create a new KeyLines chart
      klChartService.create(id, function (component) {
        // this chart reference is a standard KeyLines chart and is used internally in this directive only (to avoid recursion on klChart)
        scope.chart = component;
        // this reference is for a users' controllers (a wrapped instance of KeyLines)
        scope.klChart = klChartService.wrapObject(component, updateIsolateScope);
        /*
         * WARNING: This is a bit of a nasty caveat
         * We must broadcast the ready event only after scope.klChart has been setup
         * Angular sets up scope.klChart sometime after our assignment in another digest cycle
         * We must wait for that previous digest cycle to finish before we broadcast the ready event,
         * that way we can be sure that any listeners on the ready event will have access to the scope.klChart
         * $timeout is the way to run this after the current digest cycle (queued for when it is available)
         */
        $timeout(function () {
          // This digest call is necessary to make flash work (can be removed if you only want to support canvas)
          scope.$digest();
          klChartService.broadcast('ready', id);
        });
      });

    }
  };

}]);

angular.module('keylines').
  directive('klTimebar', ['klTimebarService', '$timeout', function (klTimebarService, $timeout) {

    return {

      restrict: 'A',  

      scope: {
        klOptions: '=',
        klTimebar: '='
      },

      controller: function ($scope) {
        $scope.$watch('klOptions', function (val) {
          if ($scope.timebar && val) {
            $scope.timebar.options(val);
          }
        }, true);
      },

      link: function (scope, element, attrs) {

        var id = element[0].id;

        function updateIsolateScope () {
          // needs timeout so that the $watch in controller can run first... (try not using $timeout then clicking turnGrey in angular demo and only tb goes grey)
          $timeout(function () {
            scope.klOptions = scope.timebar.options();
          });
        }

        //the base path is where KeyLines will load both its core files (the swf files, etc) and also its assets
        // - must include an ending / slash
        klTimebarService.config(attrs.klBasePath);

        klTimebarService.create(id, function (component) {
          scope.timebar = component;
          scope.klTimebar = klTimebarService.wrapObject(component, updateIsolateScope);
          /*
           * WARNING: This is a bit of a nasty caveat
           * We must broadcast the ready event only after scope.klTimebar has been setup
           * Angular sets up scope.klTimebar sometime after our assignment in another digest cycle
           * We must wait for that previous digest cycle to finish before we broadcast the ready event,
           * that way we can be sure that any listeners on the ready event will have access to the scope.klTimebar
           * $timeout is the way to run this after the current digest cycle (queued for when it is available)
           */
          $timeout(function () {
            // This digest call is necessary to make flash work (can be removed if you only want to support canvas)
            scope.$digest();
            klTimebarService.broadcast('ready', id);
          });
        });

      }
    };
  }]);
