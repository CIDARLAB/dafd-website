var ThreeDuFPlugin =
/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = "./src/main.js");
/******/ })
/************************************************************************/
/******/ ({

/***/ "./src/main.js":
/*!*********************!*\
  !*** ./src/main.js ***!
  \*********************/
/*! exports provided: openDesign, openDesignWithParamChanges, openDesignWithDAFDParams */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"openDesign\", function() { return openDesign; });\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"openDesignWithParamChanges\", function() { return openDesignWithParamChanges; });\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"openDesignWithDAFDParams\", function() { return openDesignWithDAFDParams; });\n/**\n * Opens the design on 3DuF.org. Please ensure that the file is available on a public url\n * @param fileurl full public url for 3DuF to open the file\n */\nfunction openDesign(fileurl) {\n    let baseurl = new URL(\"https://3duf.org/\");\n    baseurl.searchParams.append(\"file\", fileurl);\n    console.log(\"DAFD Link:\", baseurl);\n    window.open(baseurl, '_blank');\n}\n\n/**\n * Opens the design on 3DuF.org with the specified component modified according to the given parameters\n * @param fileurl full public url for 3DuF to open the file\n * @param componentname component name of the component that needs to change\n * @param params Javascript object key value pairs of parameters that need to change\n */\nfunction openDesignWithParamChanges(fileurl, componentname, params) {\n    let baseurl = new URL(\"https://3duf.org/\");\n    baseurl.searchParams.append(\"file\", fileurl);\n    baseurl.searchParams.append(\"componentname\", componentname);\n    baseurl.searchParams.append('params', JSON.stringify(params));\n    console.log(\"DAFD Link:\", baseurl);\n    window.open(baseurl, '_blank');\n}\n\n/**\n * Opens the design on 3DuF.org with the specified in the DAFD mode\n * @param fileurl full public url for 3DuF to open the file\n * @param componentname component name of the component that needs to change\n * @param params Javascript object key value pairs of parameters that need to change\n */\nfunction openDesignWithDAFDParams(params) {\n    let baseurl = new URL(\"https://3duf.org/\");\n    console.log(JSON.stringify(params));\n    baseurl.searchParams.append('dafdparams', (JSON.stringify(params)));\n    console.log(\"DAFD Link:\", baseurl);\n    window.open(baseurl, '_blank');\n}//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9zcmMvbWFpbi5qcy5qcyIsInNvdXJjZXMiOlsid2VicGFjazovL1RocmVlRHVGUGx1Z2luLy4vc3JjL21haW4uanM/NTZkNyJdLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIE9wZW5zIHRoZSBkZXNpZ24gb24gM0R1Ri5vcmcuIFBsZWFzZSBlbnN1cmUgdGhhdCB0aGUgZmlsZSBpcyBhdmFpbGFibGUgb24gYSBwdWJsaWMgdXJsXG4gKiBAcGFyYW0gZmlsZXVybCBmdWxsIHB1YmxpYyB1cmwgZm9yIDNEdUYgdG8gb3BlbiB0aGUgZmlsZVxuICovXG5leHBvcnQgZnVuY3Rpb24gb3BlbkRlc2lnbihmaWxldXJsKSB7XG4gICAgbGV0IGJhc2V1cmwgPSBuZXcgVVJMKFwiaHR0cHM6Ly8zZHVmLm9yZy9cIik7XG4gICAgYmFzZXVybC5zZWFyY2hQYXJhbXMuYXBwZW5kKFwiZmlsZVwiLCBmaWxldXJsKTtcbiAgICBjb25zb2xlLmxvZyhcIkRBRkQgTGluazpcIiwgYmFzZXVybCk7XG4gICAgd2luZG93Lm9wZW4oYmFzZXVybCwgJ19ibGFuaycpO1xufVxuXG4vKipcbiAqIE9wZW5zIHRoZSBkZXNpZ24gb24gM0R1Ri5vcmcgd2l0aCB0aGUgc3BlY2lmaWVkIGNvbXBvbmVudCBtb2RpZmllZCBhY2NvcmRpbmcgdG8gdGhlIGdpdmVuIHBhcmFtZXRlcnNcbiAqIEBwYXJhbSBmaWxldXJsIGZ1bGwgcHVibGljIHVybCBmb3IgM0R1RiB0byBvcGVuIHRoZSBmaWxlXG4gKiBAcGFyYW0gY29tcG9uZW50bmFtZSBjb21wb25lbnQgbmFtZSBvZiB0aGUgY29tcG9uZW50IHRoYXQgbmVlZHMgdG8gY2hhbmdlXG4gKiBAcGFyYW0gcGFyYW1zIEphdmFzY3JpcHQgb2JqZWN0IGtleSB2YWx1ZSBwYWlycyBvZiBwYXJhbWV0ZXJzIHRoYXQgbmVlZCB0byBjaGFuZ2VcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG9wZW5EZXNpZ25XaXRoUGFyYW1DaGFuZ2VzKGZpbGV1cmwsIGNvbXBvbmVudG5hbWUsIHBhcmFtcykge1xuICAgIGxldCBiYXNldXJsID0gbmV3IFVSTChcImh0dHBzOi8vM2R1Zi5vcmcvXCIpO1xuICAgIGJhc2V1cmwuc2VhcmNoUGFyYW1zLmFwcGVuZChcImZpbGVcIiwgZmlsZXVybCk7XG4gICAgYmFzZXVybC5zZWFyY2hQYXJhbXMuYXBwZW5kKFwiY29tcG9uZW50bmFtZVwiLCBjb21wb25lbnRuYW1lKTtcbiAgICBiYXNldXJsLnNlYXJjaFBhcmFtcy5hcHBlbmQoJ3BhcmFtcycsIEpTT04uc3RyaW5naWZ5KHBhcmFtcykpO1xuICAgIGNvbnNvbGUubG9nKFwiREFGRCBMaW5rOlwiLCBiYXNldXJsKTtcbiAgICB3aW5kb3cub3BlbihiYXNldXJsLCAnX2JsYW5rJyk7XG59XG5cbi8qKlxuICogT3BlbnMgdGhlIGRlc2lnbiBvbiAzRHVGLm9yZyB3aXRoIHRoZSBzcGVjaWZpZWQgaW4gdGhlIERBRkQgbW9kZVxuICogQHBhcmFtIGZpbGV1cmwgZnVsbCBwdWJsaWMgdXJsIGZvciAzRHVGIHRvIG9wZW4gdGhlIGZpbGVcbiAqIEBwYXJhbSBjb21wb25lbnRuYW1lIGNvbXBvbmVudCBuYW1lIG9mIHRoZSBjb21wb25lbnQgdGhhdCBuZWVkcyB0byBjaGFuZ2VcbiAqIEBwYXJhbSBwYXJhbXMgSmF2YXNjcmlwdCBvYmplY3Qga2V5IHZhbHVlIHBhaXJzIG9mIHBhcmFtZXRlcnMgdGhhdCBuZWVkIHRvIGNoYW5nZVxuICovXG5leHBvcnQgZnVuY3Rpb24gb3BlbkRlc2lnbldpdGhEQUZEUGFyYW1zKHBhcmFtcykge1xuICAgIGxldCBiYXNldXJsID0gbmV3IFVSTChcImh0dHBzOi8vM2R1Zi5vcmcvXCIpO1xuICAgIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KHBhcmFtcykpO1xuICAgIGJhc2V1cmwuc2VhcmNoUGFyYW1zLmFwcGVuZCgnZGFmZHBhcmFtcycsIChKU09OLnN0cmluZ2lmeShwYXJhbXMpKSk7XG4gICAgY29uc29sZS5sb2coXCJEQUZEIExpbms6XCIsIGJhc2V1cmwpO1xuICAgIHdpbmRvdy5vcGVuKGJhc2V1cmwsICdfYmxhbmsnKTtcbn0iXSwibWFwcGluZ3MiOiJBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBIiwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///./src/main.js\n");

/***/ })

/******/ });